import segmentation_models_pytorch as smp
from torch.nn import Sequential, Sigmoid ,Softmax, AvgPool2d, Module, Upsample
from torch.optim import Adam
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryAccuracy, BinaryRecall
import pytorch_lightning as pl
from monai.losses import GeneralizedDiceLoss, FocalLoss, TverskyLoss, HausdorffDTLoss, DiceFocalLoss
from monai.data import DataLoader,Dataset,decollate_batch
from sklearn.model_selection import train_test_split

from datetime import date
from PIL import Image
import numpy as np

from pathlib import Path
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


class RiossModel(pl.LightningModule): 
    def __init__(self, lr):
        super().__init__()
        self.model = None
        self.loss = FocalLoss()
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score() 
        self.prec = BinaryPrecision()
        self.recall = BinaryRecall()
        self.jaccard = BinaryJaccardIndex()
        self.lr = lr
        self.optimizer = Adam
        self.hparams.loss_fn = str(self.loss)
        self.hparams.optimizer = str(self.optimizer)
        self.save_hyperparameters()
        self.optimize_params=None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.optimize_params, lr=self.lr, weight_decay=1e-5)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.lr, eta_min=0)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.lr, eta_min=0)
        return optimizer #[optimizer], [lr_scheduler]

    def run_metrics(self, z, y, stage):
        metrics = {
            f'{stage}_loss': self.loss(z, y),
            f'{stage}_acc': self.valid_acc(z,y),
            f'{stage}_f1_score': self.f1_score(z,y),
            f'{stage}_precision': self.prec(z,y),
            f"{stage}_recall": self.recall(z, y),
            f'{stage}_jaccard': self.jaccard(z, y)
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics[f'{stage}_loss']
        
    def training_step(self, batch):
        x, y = batch['img'], batch['label'] 
        z = self.model(x)
        return self.run_metrics(z, y, stage='train')
         
    def validation_step(self, batch):
        x, y = batch["img"], batch["label"]
        z = self(x)
        return self.run_metrics(z, y, stage='val')
    
    def on_train_epoch_end(self):
        pass

    def create_folder(self, root_folder):
        root_folder = Path(root_folder)
        folder_name = f"{self.__class__.__name__}_{self.encoder_name}"
        subfolder_name = f"input={self.input_size}_lr={self.lr:.2e}"
        folder_path = root_folder / folder_name / subfolder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    

class SMP(RiossModel):
    def __init__(
        self, 
        input_size, 
        in_channels, 
        lr, 
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        activation=None,
    ):
        super().__init__(lr)
        self.model = smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,     
            in_channels=in_channels,                 
            classes=1,
            activation=activation
        )
        # self.backbone = self.model.encoder
        # self.other_layers  = [layer for name, layer in self.model.named_children() if name != 'encoder']
        # self.optimize_params = [param for layer in self.other_layers for param in layer.parameters()]
        self.optimize_params = self.model.parameters()

    # @property    
    # def backbone(self):
    #     return self.model.encoder.requires_grad_(False)


class AutoencoderModel(SMP): 
    def __init__(
        self, 
        input_size, 
        in_channels, 
        lr, 
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        activation='sigmoid',
    ):
        super().__init__(
            input_size=input_size, 
            in_channels=in_channels, 
            lr=lr, 
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            activation=activation,
            model=False
        )
    
    def run_metrics(self, z, y, stage):
        metrics = {
            f'{stage}_loss': self.loss(z, y),
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics[f'{stage}_loss']
    
    def validation_step(self, batch):
        pass


class TransferSegformer(RiossModel):
    def __init__(
        self, 
        input_size, 
        in_channels, 
        lr, 
        achiteture_name = "nvidia/segformer-b4-finetuned-ade-512-512",
    ):
        super().__init__(lr)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            achiteture_name, 
            num_channels=in_channels,
            ignore_mismatched_sizes=True,
            num_labels=1,
            )
        self.input_size = input_size
        self.encoder_name = 'ViT'
        self.upsample = torch.nn.Upsample(scale_factor=1/4, mode='nearest')

    def training_step(self, batch):
        x, y = batch['img'], batch['label'] 
        z = self.model(x).logits
        y = self.upsample(y)
        return self.run_metrics(z, y, stage='train')
         
    def validation_step(self, batch):
        x, y = batch["img"], batch["label"]
        z = self(x).logits
        y = self.upsample(y)
        return self.run_metrics(z, y, stage='val')


