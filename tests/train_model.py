from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BatchSizeFinder, BackboneFinetuning, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers import CometLogger


import rioss_prep.callbacks as rc
import rioss_prep.data_processing.dataset as rd
import rioss_prep.rioss_models.transfer_models as tm
import rioss_prep.rioss_models.unetr_models as urm

from torch import set_float32_matmul_precision
import time
from tqdm import tqdm

set_float32_matmul_precision('medium')
import os

import json


model = tm.SMP(
    input_size=512, 
    in_channels=2, 
    lr=1e-4, 
    encoder_name="resnet50",
    activation=None,
)

# model = urm.PedroNet(
#     img_size=512,
#     lr=8e-5,
#     depths=(2, 2, 2, 2), 
#     num_heads=(3, 6, 12, 24), 
#     feature_size=24, 
#     drop_rate=0.05, 
# )

model_folder = "/mnt/camobi_2/PHMG/delete_callback_test"


early_stopping = EarlyStopping(monitor='val_loss',
                                mode="min",
                                min_delta=0.0,
                                patience=100, 
                                verbose=True,
                                check_on_train_epoch_end=False,
                                check_finite=True,
                                log_rank_zero_only=True)


model_checkpoint = ModelCheckpoint(save_top_k = 5,
                                    monitor = 'val_f1_score',
                                    mode='max',
                                    dirpath = model_folder,
                                    filename ='model-{val_loss:3f}-{val_f1_score:.3f}-{val_precision:.3f}-{val_recall:.3f}',
                                    save_on_train_epoch_end = True,
                                    # verbose=True,
                                    )

backbone_finetuning = BackboneFinetuning(
    unfreeze_backbone_at_epoch=10,
    verbose=True,
    )

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="http://127.0.0.1:6006")

logger = CSVLogger('/mnt/camobi_2/PHMG/delete_callback_test')

with open('/mnt/camobi_2/PHMG/rioss_prep/tests/api_key.json', 'r') as file:
    data = json.load(file)
api_key = data.get('key')

comet_logger = CometLogger(
    api_key=api_key,
    workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
    # save_dir="/mnt/camobi_2/PHMG/comet_log",  # Optional
    project_name="rioss",  # Optional
    # experiment_name="2_channels",  # Optional
)


trainer = Trainer(max_epochs=300, 
                    accelerator='gpu',
                    devices=[0, 1],#[0,1] or -1
                    profiler="simple",
                    # callbacks=[early_stopping, model_checkpoint], #, early_stopping, model_checkpoint],
                    log_every_n_steps=8,
                    # logger=comet_logger,
                    use_distributed_sampler=False,
                    strategy='ddp',
                    # precision="bf16-mixed",
                    )

JSON_PATH_SPLIT = "/mnt/camobi_3/new_data/train_val.json"
files_path = '/mnt/camobi_process/new_dataset'
label_order_path = '/mnt/camobi_3/new_data/label_order.json'

CLASS_WEIGHTS = {
    "oil": 80,
    "ship": 2,
    "lookalike": 80,
    "wind": 40,
    "rain": 0,
    "land": 5,
    "biofilm": 0,
    "border": 5,
    "ocean": 15
}

win_size = 2048
resize_size = 512
rioss_datamodule = rd.RiossDataModule(
    win_size=win_size,
    dataset_folder=files_path, 
    split_path=JSON_PATH_SPLIT, 
    class_weights=CLASS_WEIGHTS,
    batch_size=32,
    resize_size=resize_size,
    label_order_path=label_order_path, 
    overlap=0.5, 
    filter_min=0.015,
    num_workers=4,
    wind_speed=False,
    incident_angle_channel=True,
)

trainer.fit(
    model, 
    datamodule=rioss_datamodule,
)



