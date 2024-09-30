from datetime import date
from os.path import join,exists,expanduser, isdir
from os import listdir, mkdir, rmdir, remove, walk, path
from torch import from_numpy, no_grad, device, cuda, inference_mode
from torch.nn import Sequential, Sigmoid ,Softmax, AvgPool2d, Module, Upsample
from torch.optim import Adam
import torch
from monai.losses import DiceLoss, FocalLoss, TverskyLoss, HausdorffDTLoss, DiceFocalLoss
from monai.networks.nets import UNet, UNETR, SwinUNETR

from pytorch_lightning import LightningModule, Trainer
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryAccuracy, BinaryRecall
import matplotlib.pyplot as plt

from time import time

from pathlib import Path


import jsonargparse
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from os import listdir
from shutil import rmtree

import netCDF4 as nc
# from RiossDataset import DatasetGenerator

# from dataset.datasets import WrapperDataset
# from dataset.transforms import ComposedTransform, ToBinaryTransformd, LabelEncodeTransform, AddAxisTransform

import segmentation_models_pytorch as smp

label_num_to_name = {
    -1: 'null',
    0: 'oil',
    1: 'ocean',
    2: 'land',
    3: 'ship',
    4: 'biofilm',
    5: 'wind',
    6: 'rain',
    7: 'dark-ocean',
}

label_name_to_num = {v: k for k, v in label_num_to_name.items()}

data_dir = Path("/home/los/david/Jupyter/Algoritmos/Notebook/createDatasetMarcos/datasets")
val_split_ratio = 1/2

data_paths = sorted(data_dir.glob("*.nc"))
num_paths = len(data_paths)
val_split_index = num_paths - int(num_paths * 1/4)
train_paths = data_paths[:val_split_index]
val_paths = data_paths[val_split_index:]
data_paths = {
    "train": train_paths,
    "val": val_paths,
}

labels = list(label_name_to_num.keys())


class PedroNet(LightningModule): #out_channels = numero de classes
    def __init__(self,
                 img_size,
                 lr,
                 depths=(2, 2, 2, 2), 
                 num_heads=(3, 6, 12, 24), 
                 feature_size=24, 
                 norm_name='instance', 
                 drop_rate=0.0, 
                 attn_drop_rate=0.0, 
                 dropout_path_rate=0.0, 
                 normalize=True, 
                 use_checkpoint=False, 
                 downsample='merging', 
                 use_v2=False,
                 window_shape=None,
                 ):
        super().__init__()
        self.model = Sequential(SwinUNETR(spatial_dims=2,
                                    in_channels=1,
                                    out_channels=1,
                                    depths=depths,
                                    img_size=img_size,
                                    feature_size=feature_size,
                                    drop_rate=drop_rate,
                                    num_heads=num_heads,
                                    norm_name=norm_name,
                                    attn_drop_rate=attn_drop_rate,
                                    dropout_path_rate=dropout_path_rate,
                                    normalize=normalize,
                                    use_checkpoint=use_checkpoint,
                                    downsample=downsample,
                                    use_v2=use_v2
                                    ))

        self.loss = FocalLoss()
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score() 
        self.prec = BinaryPrecision()
        self.recall = BinaryRecall()
        self.jaccard = BinaryJaccardIndex()
        self.lr = lr
        self.drop = drop_rate
        self.depths = depths
        self.attn_drop_rate = attn_drop_rate
        self.save_hyperparameters()

        self.img_size = img_size
        self.batch_size = False
        self.batch_size_inference = 24
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.window_shape = window_shape if window_shape else (img_size, img_size)

        self.images_path = None
        self.inference_path = None
        self.inference_num = 0

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
        
    def training_step(self,train_batch):
        x, y = train_batch['img'], train_batch['label'] 

        #forward pass
        z = self.model(x)
        loss = self.loss(z, y)  
        t_acc = self.train_acc(z, y)
        f1_score = self.f1_score(z, y)
        precision = self.prec(z, y)
        jaccard = self.jaccard(z, y)
        recall = self.recall(z, y)
        self.log('train_loss',loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=True, sync_dist=True)
        self.log('train_acc', t_acc, on_step=False, on_epoch =True, prog_bar=True, sync_dist=True)
        self.log('train_f1_score', f1_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch):
        x, y = val_batch["img"], val_batch["label"]
        z = self(x)
        loss = self.loss(z, y)
        v_acc = self.valid_acc(z,y)
        val_f1_score = self.f1_score(z,y)
        val_precision = self.prec(z,y)
        recall = self.recall(z, y)
        jaccard = self.jaccard(z, y)
        self.log('val_loss',loss, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_acc', v_acc, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_f1_score', val_f1_score, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_precision', val_precision, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end_1(self):
        if self.trainer.global_rank == 0:
            print("Realizando inferência por epoch")
            sat_img_path = "/mnt/camobi_3/new_data/images_nc/76C3.nc"
            sat_img_path = nc.Dataset(sat_img_path, 'r')
            sat_img = torch.tensor(sat_img_path.variables['Sigma0_VV_db'][:][3000:7000, 12750:15000], dtype=torch.float32)
        
            sat_img = sat_img.unsqueeze(0).unsqueeze(0)
            sat_img = sat_img.to("cuda:0")
        
            self.model.eval()
            with inference_mode():
                logits_outputs = sliding_window_inference(sat_img, 
                                    roi_size=(self.img_size),
                                    sw_batch_size=self.batch_size_inference, 
                                    predictor=self.model.to("cuda:0"), 
                                    mode='gaussian',
                                    overlap=0.7,
                                    progress=True
                                    )
                
                sigmoid_fn = Sigmoid()
                pred_outputs = sigmoid_fn(logits_outputs)
                pred_outputs = pred_outputs.to("cpu")
        
            pred_outputs = pred_outputs.squeeze()
        
            plt.imshow(pred_outputs)
            plt.show()
            self.inference_num += 1

    def save_model_dir(self):
        model_number = 0
        today = date.today()
        dirpath = expanduser('/mnt/camobi_2/PHMG/swin_unetr_models')

        for item in listdir(dirpath):
            if "Model_" in item:
                if not any(saved_model.endswith(".ckpt") for saved_model in listdir(join(dirpath, item))):
                    rmtree(join(dirpath, item))

        model_dir = self.get_folder_name()
        if not exists(model_dir):
            mkdir(model_dir)
            mkdir(join(model_dir,'images'))
            mkdir(join(model_dir,'inference'))

            self.images_path = join(model_dir,'images')
            self.inference_path = join(model_dir,'inference')

            model_dir_image = join(model_dir,'images')
            print('O diretório do Model foi criado!!')
            print(f'Caminho:{model_dir}')
            print(f'Caminho do diretório das imagens:{model_dir_image}')
        else:
            print("Modelo já foi rodado")
            model_dir = False

        return model_dir
    
    def setup_2(self, stage=None):
        dataset_generators = {
            split: DatasetGenerator(
                paths,
                input_name="sigma0",
                label_name="label",
                label_num_to_name=label_num_to_name,
                cache_dir=self.cache_dir,
                renew_cache=False,
            )
            for split, paths in data_paths.items()
        }
        
        label_datasets = {
            split: dataset_generator.generate_label_datasets(window_shape=self.window_shape, task="segmentation")
            for split, dataset_generator in dataset_generators.items()
        }

        non_empty_labels = {
            split: [label for label in labels if label in label_dataset]
            for split, label_dataset in label_datasets.items()
        }

        # label_probas: oil proba = 1/2 and other labels have uniform probas
        oil_proba = 0.5
        non_oil_classes = {
            split: [label for label in labels if label != "oil"]
            for split, labels in non_empty_labels.items()
        }
        
        non_oil_proba = {
            split: (1 - oil_proba) / len(non_oil_classes_for_split)
            for split, non_oil_classes_for_split in non_oil_classes.items()
        }
        
        label_probas = {
            split: {
                label: (oil_proba if label == "oil" else non_oil_proba[split])
                for label in labels
            }
            for split, labels in non_empty_labels.items()
        }

        # multi-class
        used_labels = list(label_probas["train"])
        label_name_to_num_enc = {v: k for k, v in enumerate(used_labels)}
        val_only_labels = [label for label in non_empty_labels["val"] if label not in non_empty_labels["train"]] 
        val_only_labels_enc = {label: -1 for label in val_only_labels}
        label_name_to_num_enc.update(val_only_labels_enc)

        # label_name_to_num_enc = {
        #     label: (0 if label == "oil" else 1)
        #     for label in set(*list(labels) for labels in non_empty_labels.values())}
        # }
        # #non_oil_to_one = {label:  for label in ....train_test_union...}

        raw_datasets_and_samplers = {
            split: dataset_generator.generate_full_dataset_and_weighted_sampler(label_datasets[split], label_probas[split])
            for split, dataset_generator in dataset_generators.items()
        }

        raw_datasets = {split: ds for split, (ds, sampler) in raw_datasets_and_samplers.items()} 
        self.samplers = {split: sampler for split, (ds, sampler) in raw_datasets_and_samplers.items()}
        self.datasets = {
            split: WrapperDataset(
                raw_dataset,
                transform=ComposedTransform([
                    LabelEncodeTransform(key="label", label_to_num=label_name_to_num, label_to_enc=label_name_to_num_enc),
                ]),
            )
            for split, raw_dataset in raw_datasets.items()
        }
    
    def train_dataloader(self):
        split = "train"
        return torch.utils.data.DataLoader(
            dataset=self.datasets[split],
            sampler=self.samplers[split],
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        split = "val"
        return torch.utils.data.DataLoader(
            dataset=self.datasets[split],
            sampler=self.samplers[split],
            batch_size=self.batch_size,
            num_workers=8,
        )
    
    
class RiossMetaUnet(LightningModule): #out_channels = numero de classes
    def __init__(self, img_size, patch_model,
                 lr,
                 depths=(2, 2, 2, 2), 
                 num_heads=(3, 6, 12, 24), 
                 feature_size=24, 
                 norm_name='instance', 
                 drop_rate=0.0, 
                 attn_drop_rate=0.0, 
                 dropout_path_rate=0.0, 
                 normalize=True, 
                 use_checkpoint=False, 
                 downsample='merging', 
                 use_v2=False 
                 ):
        super().__init__()
        self.patch_model = patch_model
        self.model = Sequential(SwinUNETR(spatial_dims=2,
                                    in_channels=2,
                                    out_channels=1,
                                    depths=depths,
                                    img_size=img_size,
                                    feature_size=feature_size,
                                    drop_rate=drop_rate,
                                    num_heads=num_heads,
                                    norm_name=norm_name,
                                    attn_drop_rate=attn_drop_rate,
                                    dropout_path_rate=dropout_path_rate,
                                    normalize=normalize,
                                    use_checkpoint=use_checkpoint,
                                    downsample=downsample,
                                    use_v2=use_v2
                                    ))    

        # self.model = smp.Unet(
        #     encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=1,
        #     # decoder_use_batchnorm=True,                   # model output channels (number of classes in your dataset)
        # )

        self.dataset_img_size = 2048
        self.img_size = img_size
        self.loss = FocalLoss()
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score() 
        self.prec = BinaryPrecision()
        self.recall = BinaryRecall()
        self.jaccard = BinaryJaccardIndex()
        self.lr = lr
        self.drop = drop_rate
        self.depths = depths
        self.attn_drop_rate = attn_drop_rate
        #self.save_hyperparameters()
        
        self.unfold_fn = torch.nn.Unfold(kernel_size=(self.img_size, self.img_size), stride=(self.img_size, self.img_size))
        self.fold_fn = torch.nn.Fold(kernel_size=(self.img_size, self.img_size), output_size=(self.dataset_img_size, self.dataset_img_size), stride=(self.img_size, self.img_size))
        self.upsample_fn = Upsample(scale_factor=self.img_size/self.dataset_img_size, mode='nearest')

        
        self.batch_size = 8
        self.batch_size_inference = 24
        self.num_heads = num_heads
        self.feature_size = feature_size
        #self.automatic_optimization = False

        self.images_path = None
        self.inference_path = None
        self.inference_num = 0

    def create_folder(self, root_folder):
        root_folder = Path(root_folder)
        folder_name = f"{self.__class__.__name__}_input={self.img_size}_lr={self.lr:.2e}"
        folder_path = root_folder / folder_name

        folder_path.mkdir(parents=True, exist_ok=True)
        inference_path = folder_path / 'inferences'
        inference_path.mkdir(exist_ok=True)

        return folder_path

    def forward_(self, x):
        return self.model(self.upsample_fn(x))      

    def forward_(self, x):
        batch_size, channels, height, width = x.shape
        
    
        x = self.unfold_fn(x)
        extra_batch = x.shape[-1]
        x = x.view(batch_size, channels, self.img_size, self.img_size, -1).permute(0, 4, 1, 2, 3)
        x = x.reshape(-1, 1, self.img_size, self.img_size)
        

        x1 = self.patch_model(x)
        x1 = Sigmoid()(x1)
        
        x = x.reshape(batch_size, extra_batch, channels, self.img_size, self.img_size)
        x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, extra_batch)
        x = self.fold_fn(x)
        
        x1 = x1.reshape(batch_size, extra_batch, channels, self.img_size, self.img_size)
        x1 = x1.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, extra_batch)
        x1 = self.fold_fn(x1)
        
        x = self.upsample_fn(x)
        x1 = self.upsample_fn(x1)
        
        return self.model(torch.cat((x, x1), dim=1))

    def forward(self, x):
        x_one_channel = x[:, 0].unsqueeze(1)
        batch_size, channels, height, width = x_one_channel.shape
        
    
        x_one_channel = self.unfold_fn(x_one_channel)
        extra_batch = x_one_channel.shape[-1]
        x_one_channel = x_one_channel.view(batch_size, channels, self.img_size, self.img_size, -1).permute(0, 4, 1, 2, 3)
        x_one_channel = x_one_channel.reshape(-1, 1, self.img_size, self.img_size)
        

        x1 = self.patch_model(x_one_channel)
        x1 = Sigmoid()(x1)
        
        # x = x.reshape(batch_size, extra_batch, channels, self.img_size, self.img_size)
        # x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, extra_batch)
        # x = self.fold_fn(x)
        
        x1 = x1.reshape(batch_size, extra_batch, channels, self.img_size, self.img_size)
        x1 = x1.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, extra_batch)
        x1 = self.fold_fn(x1)
        
        x = self.upsample_fn(x)
        x1 = self.upsample_fn(x1)
        
        return self.model(torch.cat((x1, x), dim=1))
        
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
        
    def training_step(self,train_batch):
        x, y = train_batch['img'], train_batch['label'] 
        #forward pass
        z = self(x)
        y = self.upsample_fn(y)
        loss = self.loss(z, y)
        t_acc = self.train_acc(z, y)
        f1_score = self.f1_score(z, y)
        precision = self.prec(z, y)
        jaccard = self.jaccard(z, y)
        recall = self.recall(z, y)

        self.log('train_loss',loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=True, sync_dist=True)
        self.log('train_acc', t_acc, on_step=False, on_epoch =True, prog_bar=True, sync_dist=True)
        self.log('train_f1_score', f1_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch):
        x, y = val_batch["img"], val_batch["label"]
        z = self(x)
        y = self.upsample_fn(y)
        loss = self.loss(z, y)
        
        v_acc = self.valid_acc(z,y)
        val_f1_score = self.f1_score(z,y)
        val_precision = self.prec(z,y)
        recall = self.recall(z, y)
        jaccard = self.jaccard(z, y)

        self.log('val_loss',loss, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_acc', v_acc, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_f1_score', val_f1_score, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log('val_precision', val_precision, on_step=False, on_epoch = True, prog_bar = True, sync_dist=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_jaccard', jaccard, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss