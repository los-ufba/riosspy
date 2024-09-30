from multiprocessing import Process
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import re
import warnings

import cv2
import json

from subprocess import run, DEVNULL

from monai.inferers import sliding_window_inference

from dask.diagnostics import ProgressBar

from zipfile import ZipFile
from zipfile import BadZipFile
from zipfile import is_zipfile

import xmltodict

import time

from pathlib import Path

from scipy.interpolate import griddata
from skimage.transform import resize


class SentinelProduct:
    def __init__(self, file_path):
        self.product_path = Path(file_path)
        self.name = self.product_path.stem.split('.')[0]
        self.simple_name = self.name[-4:]

    def __str__(self):
        return self.simple_name

    def unzip(self, unzip_folder):
        unzip_folder = Path(unzip_folder)
        assert unzip_folder.exists(), f"Folder: '{unzip_folder}' does not exist!"
        
        print("Unziping:", self.name)
        
        with ZipFile(self.product_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, unzip_folder)
                except BadZipFile:
                    print(f"Imposible to unzip: {self.name}. file is incomplete or corrupted!")
        self.product_path = (unzip_folder / self.name).with_suffix('.SAFE')

    def change_xmlfile(self, netcdf_folder, xml_path):
        netcdf_folder = Path(netcdf_folder)
        with open(xml_path) as file:
            json_data = xmltodict.parse(file.read())

        netcdf_path =  (netcdf_folder / self.simple_name).with_suffix('.nc')
        json_data['graph']['node'][0]['parameters']['file'] = self.product_path
        json_data['graph']['node'][-1]['parameters']['file'] = netcdf_path

        print(json_data['graph']['node'][0]['parameters']['file'])
        with open(xml_path, 'w') as file:
            file.write(xmltodict.unparse(json_data, pretty=True))

    def save_nc(self, netcdf_folder, gpt_path, xml_path):
        self.change_xmlfile(netcdf_folder, xml_path)
        shell = run([gpt_path, xml_path])#, stdout=DEVNULL, stderr=DEVNULL)

class DataAssembler:
    def __init__(self, name, dir):
        self.dir = Path(dir)
        self.name = name
        self.nc_data = (self.dir / 'nc_data' / self.name).with_suffix('.nc')
        self.label_json_path = (self.dir / 'labels_json' / self.name).with_suffix('.json')
        self.png_sigma0 = (self.dir / 'png_sigma0' / self.name).with_suffix('.png')
        self.png_label = (self.dir / 'png_label' / self.name).with_suffix('.png')
        self.no_land = (Path("/mnt/camobi_2/PHMG/Sentinel_Acquisition") / 'no_land' / self.name).with_suffix('.json')
        self.ocn_path = (self.dir / 'ocn_nc' / self.name).with_suffix('.nc')
        self.label_order_path = self.dir / 'label_order.json'

        with open(self.label_order_path, 'r') as file:
            self.label_order = json.load(file)

        self.label_name_to_num = self.label_order['label_name_to_num']
        self.label_aliases = self.label_order['label_aliases']

        def __str__(self):
            return json.dumps({
                "name": self.name,
                "dir": str(self.dir),
                "nc_data": str(self.nc_data),
                "label_json_path": str(self.label_json_path),
                "png_sigma0": str(self.png_sigma0),
                "png_label": str(self.png_label),
                "no_land": str(self.no_land),
                "label_order_path": str(self.label_order_path),
                "label_name_to_num": self.label_name_to_num,
                "label_aliases": self.label_aliases
            }, indent=4)
    
    def netcdf(self):
        return xr.open_dataset(self.nc_data)
    
    def sigma_zero(self, sigma0='Sigma0_VV_db'):
        return self.netcdf()[sigma0]
    
    def save_img(path, matrix):
        matrix = np.array(matrix)
        normalize_png = ((matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))) * 255
        cv2.imwrite(str(path), normalize_png)

    def save_sigma0_png(self, overwite=False):
        if self.png_sigma0.exists() or overwite:
            DataAssembler.save_img(self.png_sigma0, self.sigma_zero())
    
    def save_label_png(self, overwite=False):
        if self.png_label.exists() or overwite:
            DataAssembler.save_img(self.png_label, self.label())

    def label(self, dtype=np.int8):
        dict_patches = {key: [] for key in self.label_name_to_num.keys()}

        with self.label_json_path.open() as file:
            label_json = json.load(file)

        with self.no_land.open() as file:
            no_land_json = json.load(file)

        shapes = [*label_json['shapes'], *no_land_json['shapes']]

        mask_array = np.full((label_json['imageHeight'], label_json['imageWidth']), self.label_name_to_num['ocean'], dtype=dtype)


        for shape in shapes:
            shape_label = re.sub(r"[^a-zA-Z0-9]", "", shape['label'].lower())
            for label, aliases in self.label_aliases.items():
                if not shape_label in aliases:
                    continue
                points = np.array(shape['points'], dtype='int32')
                dict_patches[label].append(points)
                break
            else:
                warnings.warn(f'INVALID LABEL FOUND: {shape_label} IN FILE: {self.label_json_path.name}')
        
        for label, polygons in dict_patches.items():
            mask_array = cv2.fillPoly(mask_array, polygons, color=self.label_name_to_num[label])
            
        nan_mask = np.isnan(self.sigma_zero())
        mask_array = np.where(nan_mask, -1, mask_array)
        return mask_array
    
    def create_polygons(self):
        edited_contours = []
        binary_image = self.label
        contours, hierarchy = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for poly in contours:
            if len(poly) > 50:
                approx = cv2.approxPolyDP(poly, 0.8, True)
                approx = np.squeeze(approx)
                edited_contours.append(approx)
        return edited_contours
    
    def extract_values(self, sample_gap):
        sample_height = sample_gap[0]
        sample_width = sample_gap[1]

        label_matrix = self.label()
        extracted_values = label_matrix[:-sample_height:sample_height, :-sample_width:sample_width]
        rows, cols = np.meshgrid(
            np.arange(0, label_matrix.shape[0] - sample_height - 1, sample_height).astype(np.uint16), 
            np.arange(0, label_matrix.shape[1] - sample_height - 1, sample_width).astype(np.uint16)
        )
        coordinates = np.stack((rows, cols), axis=-1).flatten()
        return extracted_values, coordinates
    
    def setup_ocn_file(self, variable):
        if self.ocn_path.exists():
            ocn_file = xr.open_dataset(self.ocn_path)
            variable = ocn_file[variable].rename({"owiAzSize": "y", "owiRaSize": "x"})
            wind_speed_lat = ocn_file.owiLat
            wind_speed_lon = ocn_file.owiLon

            variable = variable.assign_coords(
                lat=(('y', 'x'), wind_speed_lat.values),
                lon=(('y', 'x'), wind_speed_lon.values)
            )
            return variable

    
    def interpolate_ocn_file(self, data_array, factor=4):
        nc_data = xr.open_dataset(self.nc_data)

        wind_speed_lat = data_array.lat.values
        wind_speed_lon = data_array.lon.values

        factor = 4
        sigma0_lat = nc_data.lat.values[::factor,::factor]
        sigma0_lon = nc_data.lon.values[::factor,::factor]

        interpolated_data = griddata(
            (wind_speed_lat.ravel(), wind_speed_lon.ravel()),
            data_array.values.ravel(),
            (sigma0_lat, sigma0_lon),
            fill_value=np.nan,
            method='linear',
            rescale=True,
        )

        output_grid = resize(
            interpolated_data, 
            nc_data.lat.shape,
            order=1,
            mode='constant',
            cval=np.nan,
            anti_aliasing=False
        )

        return output_grid




    def to_nc_dataset(self, output_dir, sample_gap=(8, 8), wind_speed=False):
        output_dir = Path(output_dir)
        assert self.nc_data.exists(), f"NetCdf data: \"{self.nc_data}\" does not exist!"
        assert self.label_json_path.exists(), f"Json Path: \"{self.label_json_path}\" does not exist!"
        
        output_path = output_dir / self.nc_data.name
        output_tmp_path = output_path.with_suffix(".nc.tmp")
        output_dir.mkdir(exist_ok=True)

        if output_path.exists():
            sigma0_mod_time = self.nc_data.stat().st_mtime
            labelme_mod_time = self.label_json_path.stat().st_mtime
            output_mod_time = output_path.stat().st_mtime
            if sigma0_mod_time < output_mod_time and labelme_mod_time < output_mod_time:
                warnings.warn(f"{self.name} is up to date")
                return
            
        ds = None
        for var_name in self.netcdf().data_vars:
            var_name_lower = var_name.lower()
            if "sigma" in var_name_lower and "0" in var_name_lower:
                ds = self.netcdf().rename({var_name: "sigma0"})
                break
        
        if ds is None:
            warnings.warn(f"Sigma0 variable was not found in {self.nc_data}")
            return
        
        ds = ds[["sigma0", "incident_angle"]]

        # extract and add labels
        dims = ds["sigma0"].dims
        ds["label"] = dims, self.label()

        dims_div = ('y_div', 'x_div')
        extracted_values = self.extract_values(sample_gap=(8,8))[0]
        data_array = xr.DataArray(extracted_values, dims=dims_div)

        ds.coords['y_div'] = np.arange(extracted_values.shape[0])
        ds.coords['x_div'] = np.arange(extracted_values.shape[1])
        ds['extracted_values'] = data_array

        sigma0_arr = ds.sigma0.values
        ds['sigma0_min'] = sigma0_arr.min()
        ds['sigma0_max'] = sigma0_arr.max()
        ds['sigma0_std'] = sigma0_arr.std()
        ds['sigma0_mean'] = sigma0_arr.mean()
        ds['sample_gap'] = sample_gap[0]

        if wind_speed:
            if self.ocn_path.exists():
                wind_speed_dataarray = self.setup_ocn_file('owiEcmwfWindSpeed')
                interpol_wind_speed = self.interpolate_ocn_file(wind_speed_dataarray)
            else:
                nc_data = xr.open_dataset(self.nc_data)
                interpol_wind_speed = np.full(nc_data.lat.shape, np.float32(0))

            mask = np.isnan(sigma0_arr)
            interpol_wind_speed[mask] = np.nan

            ds["wind_speed"] = (ds.sigma0.dims, interpol_wind_speed)


        write_job = ds.to_netcdf(output_tmp_path, compute=False)
        with ProgressBar():
            print(f"Writing to {output_path}")
            write_job.compute()
        output_tmp_path.rename(output_path)


if __name__ == '__main__':
    RAW_FOLDER = Path('/mnt/camobi_2/PHMG/Sentinel_Acquisition/ospo_noaa_data/products')
    NC_FOLDER = Path('/mnt/camobi_3/new_data/nc_data')
    PATH_TO_GPT = "/home/camobi/snap/bin/gpt"
    SAR_TO_NC_GRAPH = "/mnt/camobi_2/PHMG/Sentinel_Acquisition/graphs/ZIP_to_NC.xml"
    UNZIP_FOLDR = Path('/mnt/camobi_2/PHMG/Sentinel_Acquisition/ospo_noaa_data/unzip_folder')
    DATASET_FOLDER = Path('/mnt/camobi_process/new_dataset')
    # rever o fill value para dados que não possuem vento ou sua borda
    output_dir = Path('/mnt/camobi_process/test_dataset')
    nc_dir = Path('/mnt/camobi_3/new_data/nc_data')
    for file in tqdm(nc_dir.glob('*.nc')):
        label_json = (nc_dir.parent / 'labels_json' / file.stem).with_suffix('.json')
        no_land_json = (Path('/mnt/camobi_2/PHMG/Sentinel_Acquisition') / 'no_land' / file.stem).with_suffix('.json')
        # print(label_json, no_land_json)
        if (label_json.exists() and no_land_json.exists()):
            ds = DataAssembler(file.stem, '/mnt/camobi_3/new_data')
            ds.to_nc_dataset(DATASET_FOLDER)


    # for safe_zip in UNZIP_FOLDR.glob('*.SAFE'):
    #     duplicated = False
    #     nc_folder_list = list(NC_FOLDER.iterdir())
    #     for nc_path in NC_FOLDER.iterdir():
    #         if nc_path.stem == safe_zip.stem[-4:]:
    #             duplicated = True

    #     if not duplicated:
    #         print(nc_path.stem)
    #         sentinel_product = SentinelProduct(safe_zip)
    #         sentinel_product.save_nc(NC_FOLDER, PATH_TO_GPT, SAR_TO_NC_GRAPH)
    #         # try:
    #         #     sentinel_product.unzip(UNZIP_FOLDR)
    #         # except BadZipFile:
    #         #     pass




