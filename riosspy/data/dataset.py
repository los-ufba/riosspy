import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
import torch
from torch import tensor
from torchvision.transforms import Compose
from monai.transforms import RandRotate90d, RandAxisFlipd, SignalFillEmptyd, ToTensord, LoadImaged, Resized
from monai.inferers import sliding_window_inference
from torch import from_numpy, no_grad, device, cuda, inference_mode
from torch.nn import Sequential, Sigmoid

from functools import lru_cache
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from flatten_dict import flatten
from itertools import chain

from tqdm import tqdm
from os.path import join
import json
import time
import random
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.stride_tricks import as_strided
import warnings
from collections import Counter


import cv2

from time import time

from pathlib import Path

import riosspy.data.transforms as rt

class RiossDataModule(pl.LightningDataModule):
    def __init__(self, 
                 win_size, 
                 dataset_folder, 
                 split_path, 
                 class_weights, 
                 batch_size, 
                 label_order_path, 
                 resize=None, 
                 overlap=0, 
                 filter_min=0,
                 filter_max=1,
                 wind_speed=False,
                 incident_angle_channel=False,
                 num_workers=0,
                 transforms=None, 
                 seed=42):
        super().__init__()
        self.win_size = win_size
        self.overlap = overlap

        self.resize = (win_size, win_size)
        if not resize == None:
            self.resize = (resize, resize)

        self.dataset_folder = Path(dataset_folder)
        self.label_order_path = label_order_path
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.generator = torch.Generator().manual_seed(seed)
        self.filter_max = filter_max
        self.filter_min = filter_min
        self.wind_speed = wind_speed
        self.incident_angle_channel = incident_angle_channel
        
        if self.transforms == None:
            self.transforms = Compose([
                ToTensord(keys=["img"], dtype=torch.float),
                ToTensord(keys=["label"], dtype=torch.uint8),
                rt.ToBinaryTransformd(keys=["label"], label_num=1),
                SignalFillEmptyd(keys=["img", "label"], replacement=0.0),
                rt.EnsureChanellFirstd(keys=["img", "label"]),
                Resized(keys=["img", "label"], spatial_size=self.resize, mode=['nearest', 'nearest']),
                RandAxisFlipd(keys=["img", "label"], prob=2/3),
                RandRotate90d(keys=["img", "label"], prob=3/4)
            ])

        with open(split_path, 'r') as json_file:
            train_val_dict = json.load(json_file)
        self.train_val_dict = train_val_dict

        self.hparams.val_dict = self.train_val_dict['val']
        self.hparams.transf = str(self.transforms)
        self.hparams.data_module_name = self.__class__.__name__
        self.save_hyperparameters()


    def setup(self, stage=None):
        data_splited = self.split_data()

        params = {
            "win_size": self.win_size, 
            "class_weights": self.class_weights, 
            "transforms": self.transforms, 
            "overlap": self.overlap, 
            "label_order_path": self.label_order_path,
            "filter_max": self.filter_max,
            "filter_min": self.filter_min,
            "wind_speed": self.wind_speed,
            "incident_angle_channel": self.incident_angle_channel,

        }
        
        self.val_ds = RiossDataset(data_splited['val'], **params)
        self.train_ds = RiossDataset(data_splited['train'], **params)
        self.test_ds = RiossDataset(data_splited['test'], **params)


        print(self.train_ds.classes_len())
        print(self.val_ds.classes_len())
        print(self.test_ds.classes_len())

        if self.trainer:
            self.batch_size_per_device = self.batch_size // self.trainer.num_devices

    def train_dataloader(self):
        sampler = RiossSampler(self.train_ds, self.train_ds.weighted_arr() ,generator=None)
        batchsampler = BatchSampler(sampler, batch_size=self.batch_size_per_device, drop_last=False)
        return DataLoader(self.train_ds, batch_sampler=batchsampler, num_workers=self.num_workers)

    def val_dataloader(self):
        sampler = RiossSampler(self.val_ds, self.val_ds.weighted_arr() ,generator=self.generator)
        batchsampler = BatchSampler(sampler, batch_size=self.batch_size_per_device, drop_last=False)
        return DataLoader(self.val_ds, batch_sampler=batchsampler, num_workers=self.num_workers)

    def test_dataloader(self):
        sampler = RiossSampler(self.test_ds, self.test_ds.weighted_arr(), generator=self.generator)
        batchsampler = BatchSampler(sampler, batch_size=self.batch_size_per_device, drop_last=False)
        return DataLoader(self.val_ds, batch_sampler=batchsampler, num_workers=self.num_workers)
  
    def split_data(self):
        output = {'train':[], 'val': [], 'test': []}
        for filepath in self.dataset_folder.glob('*.nc'):
            if filepath.stem in self.train_val_dict['val']:
                output['val'].append(filepath)
            elif filepath.stem in self.train_val_dict['test']:
                output['test'].append(filepath)
            else:
                output['train'].append(filepath)
        return output


class NetcdfDataset:
    def __init__(self, file_path, win_size, overlap=0, label_order_path=None, filter_max=1, filter_min=0):
        self.file_path = Path(file_path)
        self.name = self.file_path.name
        self.win_size = win_size
        self.label_order_path = label_order_path
        self.xarray_ds = xr.open_dataset(self.file_path)
        self.label = self.xarray_ds.label
        self.sigma0 = self.xarray_ds.sigma0
        self.sample_gap = self.xarray_ds.sample_gap
        self.filter_max = filter_max
        self.filter_min = filter_min
        self.class_coords_dict = None
        self.wind_speed = self.xarray_ds.wind_speed

        assert overlap >= 0 and overlap < 1, f'Overlap must be betwen 0 and 1'
        self.overlap = overlap
        
        with open(self.label_order_path, 'r') as file:
            self.label_order = json.load(file)
        self.label_name_to_num = self.label_order['label_name_to_num']
        self.label_aliases = self.label_order['label_aliases']

    def __repr__(self):
        return self.file_path.name

    def flatten_patches(self):
        np_array = self.xarray_ds.extracted_values.values
        patch_size = self.win_size // int(self.sample_gap)
        step_size = int(patch_size * (1 - self.overlap))
  
        patch_size = np.full(2, patch_size)
        step_size = np.full(2, step_size)
        
        new_shape = (np_array.shape - patch_size) // step_size + 1
        new_shape = np.concatenate((new_shape, patch_size), axis=0)
        
        arr_strides = np.array(np_array.strides)
        new_strides = np.concatenate((arr_strides * step_size, arr_strides), axis=0)
        
        patches = as_strided(np_array, shape=new_shape, strides=new_strides)
        return patches.reshape(-1, *patch_size)
    
    def grid_coords(self):
        np_array = self.xarray_ds.extracted_values.values
        patch_size = self.win_size // int(self.sample_gap)
        step_size = int(patch_size * (1 - self.overlap))

        num_patches_y = (np_array.shape[0] - patch_size) // step_size + 1
        num_patches_x = (np_array.shape[1] - patch_size) // step_size + 1

        base_sequence_rows = np.arange(num_patches_y).astype(np.int16)
        base_sequence_cols = np.arange(num_patches_x).astype(np.int16)

        layer1 = np.tile(base_sequence_rows.reshape(-1, 1), (1, num_patches_x))
        layer2 = np.tile(base_sequence_cols, (num_patches_y, 1))

        matrix_coords = np.stack([layer1, layer2], axis=-1).reshape(-1, 2)

        return matrix_coords * int(self.win_size * (1 - self.overlap))
    

    def _class_coords(self):
        patch_size = self.win_size // int(self.sample_gap)
        class_patches = {key: [] for key in self.label_name_to_num.keys()}
        max_number = patch_size ** 2
        flatten_patches = self.flatten_patches()
        grid_coords = self.grid_coords()
        
        for key, value in self.label_name_to_num.items():
            # contains_value = np.any(patches == value, axis=(1, 2))
            # é possível implementar o peso por píxel
            total_sum = np.sum(flatten_patches == value, axis=(1, 2)) / max_number
            filter_arr = np.logical_and(total_sum > self.filter_min, total_sum < self.filter_max)
            class_patches[key] = grid_coords[filter_arr]
        return class_patches
    
    def class_coords(self):
        if not self.class_coords_dict == None:
            return self.class_coords_dict
        self.class_coords_dict = self._class_coords()
        return self.class_coords_dict
    
    def class_count(self):
        output = {}
        for class_name, coords_list in self.class_coords().items():
            output[class_name] = len(coords_list)
        return output

class RiossDataset(Dataset):
    def __init__(self, list_paths, win_size, class_weights, transforms, label_order_path, overlap=None, filter_max=1, filter_min=0, wind_speed=False, incident_angle_channel=False):
        self.father_path = Path(list_paths[0]).parent
        self.list_paths = list_paths
        self.num_paths = len(list_paths)
        self.transforms = transforms
        self.win_size = win_size
        self.overlap = overlap
        self.label_order_path = label_order_path
        self.class_weights = class_weights
        self.filter_max = filter_max
        self.filter_min = filter_min
        self.wind_speed = wind_speed
        self.incident_angle_channel = incident_angle_channel

        self.netcdf_datasets = {}
        for file_path in tqdm(self.list_paths):
            file_path = Path(file_path)
            self.netcdf_datasets[file_path.name] = NetcdfDataset(
                file_path, 
                win_size, 
                overlap=self.overlap, 
                label_order_path=self.label_order_path, 
                filter_max=self.filter_max, 
                filter_min=self.filter_min
            )

        with open(self.label_order_path, 'r') as file:
            self.label_order = json.load(file)
        self.label_name_to_num = self.label_order['label_name_to_num']
        self.label_aliases = self.label_order['label_aliases']

        self.coords_flatted = None
        self.weights_flatted = None
    
    @lru_cache(maxsize=1)
    def all_class_coords(self):
        output = {}
        for netcdf_ds in self.netcdf_datasets.values():
            output[netcdf_ds.name] = netcdf_ds.class_coords()
        return output
    
    @lru_cache(maxsize=1)
    def all_class_count(self):
        output = {}
        for netcdf_ds in self.netcdf_datasets.values():
            output[netcdf_ds.name] = netcdf_ds.class_count()
        return output

    def flatten_coords_arr(self):
        concat_coords = np.concatenate(list(flatten(self.all_class_coords()).values()))
        return concat_coords
    
    def names_extended(self):
        class_count_arr = []
        for nc_count in list(self.all_class_count().values()):
            class_count_arr.append(list(nc_count.values()))
        class_count_arr = np.array(class_count_arr)

        name_class_arr = np.array(list(flatten(self.all_class_coords()).keys()))
        names_extended = np.repeat(name_class_arr, class_count_arr.flatten(), axis=0)
        return names_extended
    
    def classes_len(self):
        output = {key: 0 for key in self.label_name_to_num.keys()}
        for netcdf_ds in self.netcdf_datasets.values():
            for class_name, count in netcdf_ds.class_count().items():
                output[class_name] += count
        return output

    @lru_cache(maxsize=1)
    def weighted_arr(self):
        class_weights_arr = np.array(list(self.class_weights.values())).astype(np.float32)
        class_len_arr = np.array(list(self.classes_len().values())).astype(np.float32)

        class_count_arr = []
        for item in list(self.all_class_count().values()):
            class_count_arr.append(list(item.values()))
        class_count_arr = np.array(class_count_arr)

        weights = np.divide(class_weights_arr, class_len_arr, out=np.zeros_like(class_weights_arr), where=(class_len_arr!=0))
        weights_extended = np.resize(weights, class_count_arr.flatten().shape)
        weighted_array = np.repeat(weights_extended, class_count_arr.flatten())
        return(weighted_array)
    
    def class_percent(self):
        output_dict = {key: 0 for key in self.label_name_to_num.keys()}
        total_value = 0
        classes_len = self.classes_len()
        for value in classes_len.values():
            total_value += value
        
        for key, value in classes_len.items():
            output_dict[key] = round(value / total_value * 100, 4)
        return output_dict
    
    def get_input_label(self, file_name, coord):
        netcdf_ds = self.netcdf_datasets[file_name]

        y_slice = slice(coord[0], coord[0] + self.win_size)
        x_slice = slice(coord[1], coord[1] + self.win_size)

        input_img = netcdf_ds.sigma0.isel(y=y_slice, x=x_slice).values
        label = netcdf_ds.label.isel(y=y_slice, x=x_slice).values

        if self.wind_speed:
            wind_speed = netcdf_ds.xarray_ds['wind_speed'].isel(y=y_slice, x=x_slice).values
            input_img = np.stack((input_img, wind_speed), axis=0)

        if self.incident_angle_channel:
            incident_angle = netcdf_ds.xarray_ds['incident_angle'].isel(y=y_slice, x=x_slice).values
            input_img = np.stack((input_img, incident_angle), axis=0)
            
        return {'img': input_img, 'label': label}
    
    def __len__(self):
        return self.classes_len()['oil']
    
    def __getitem__(self, input):
        file_name = self.names_extended()[input][0]
        coord = self.flatten_coords_arr()[input]
        output = self.get_input_label(file_name, coord)
        return self.transforms(output)
    

class RiossSampler(DistributedSampler):
    def __init__(self, dataset, weights, generator=None, names_extended=None):
        super().__init__(dataset=dataset)
        self.weights = weights
        self.generator = generator
        self.names_extended = names_extended

    def __iter__(self):
        indices = torch.multinomial(tensor(self.weights), self.num_samples * self.num_replicas, replacement=True, generator=self.generator)
        indices = indices[self.rank:self.total_size:self.num_replicas]
        yield from iter(indices.tolist())

    def ratio(self, type):
        type_coices = {
            'classes': 1,
            'files': 0
        }
        indices = np.array(list(self.__iter__()))
        class_names = self.names_extended[indices][:, type_coices[type]]
        unique_tuple = np.unique(class_names, return_counts=True)
        result_dict = {key: round(value/np.sum(unique_tuple[1]) * 100, 2) for key, value in zip(*unique_tuple)}
        return result_dict
    
    def ratio_loop(self, times, type):
        output_dict = {}
        for _ in range(times):
            for key, value in self.ratio(type).items():
                if not key in output_dict:
                    output_dict[key] = 0
                output_dict[key] += value

        for key, value in output_dict.items():
            output_dict[key] = round(value / times, 2)
        return output_dict

class RiossSimpleSampler(RiossSampler):
    def __init__(self, weights, num_samples, generator=None, names_extended=None):
        self.weights = weights
        self.num_samples = num_samples
        self.generator = generator
        self.names_extended = names_extended

    def __iter__(self):
        indices = torch.multinomial(tensor(self.weights), self.num_samples, replacement=True, generator=self.generator)
        yield from iter(indices.tolist())


class RiossAutoencoderDM(RiossDataModule):
    def __init__(self, dataset_folder, split_path, dict_weights, batch_size, num_workers=0, transforms=None, seed=42):
        super().__init__(dataset_folder=dataset_folder, 
                    split_path=split_path, 
                    dict_weights=dict_weights, 
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    transforms=transforms, 
                    seed=seed)
        
        if self.transforms == None:
            self.transforms = Compose([
                LoadImaged(keys=["img", "label"]),
                ToTensord(keys=["img"], dtype=torch.float),
                ToTensord(keys=["label"], dtype=torch.float),
                SignalFillEmptyd(keys=["img", "label"], replacement=0.0),
                RandAxisFlipd(keys=["img", "label"], prob=2/3),
                RandRotate90d(keys=["img", "label"], prob=3/4)
            ])
    
    def setup(self, stage=None):
        data_splited = self.split_data()

        self.train_ds = RiossAutoEncoderDS(data_splited['train'], self.dataset_folder, transform=self.transforms)
        self.val_ds = RiossAutoEncoderDS(data_splited['val'], self.dataset_folder, transform=self.transforms)

        print(self.train_ds.classes_len())
        print(self.val_ds.classes_len())

        self.batch_size_per_device = self.batch_size // self.trainer.num_devices


class RiossAutoEncoderDS(RiossDataset):
    def __init__(self, data, dataset_folder, transform):
        super().__init__(
            data=data, 
            dataset_folder=dataset_folder, 
            transform=transform,
        )

    def __getitem__(self, index):
        patch = self.data_patches[index]
        output = {
            'img': self.dataset_folder / 'img' / patch,
            'label': self.dataset_folder / 'img' / patch
        }
        return self.transform(output)



if __name__ == '__main__':
    JSON_PATH_SPLIT = "/mnt/camobi_3/new_data/train_val.json"
    files_path = Path('/mnt/camobi_process/test_dataset')
    file_path = files_path / 'D205.nc'
    label_order_path = '/mnt/camobi_3/new_data/label_order.json'
    
    # test = NetcdfDataset(files_path, 2048, int(2048 * 0.75), label_order_path)
    # print(test.get_sample_coords())

    CLASS_WEIGHTS = {
    "oil": 120,
    "ship": 2,
    "lookalike": 10,
    "wind": 0,
    "rain": 0,
    "land": 5,
    "biofilm": 0,
    "border": 5,
    "ocean": 15
    }
    transforms = Compose([
                ToTensord(keys=["img"], dtype=torch.float),
                ToTensord(keys=["label"], dtype=torch.uint8),
                rt.ToBinaryTransformd(keys=["label"], label_num=1),
                SignalFillEmptyd(keys=["img", "label"], replacement=0.0),
                rt.EnsureChanellFirstd(keys=["img", "label"]),
                Resized(keys=["img", "label"], spatial_size=(512, 512), mode=['nearest', 'nearest']),
                RandAxisFlipd(keys=["img", "label"], prob=2/3),
                RandRotate90d(keys=["img", "label"], prob=3/4)
    ])



    test_ds = RiossDataset(list(files_path.glob('*.nc')), 2048, CLASS_WEIGHTS, transforms=transforms, label_order_path=label_order_path, overlap=0, wind_speed=True)
    # test_sampler = TestSampler(test_ds)
    # for item in test_sampler:
    #     print(item)
    # test_dl =  DataLoader(test_ds, sampler=test_sampler)
    # print(next(iter(test_dl)))
    print(test_ds[3000]['img'].shape)

    # indices = torch.multinomial(weights, 24, replacement=False)





    

           



    


    
