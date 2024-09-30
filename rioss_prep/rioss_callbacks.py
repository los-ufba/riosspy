from lightning.pytorch.callbacks.callback import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BatchSizeFinder, BackboneFinetuning
from pytorch_lightning.loggers import CSVLogger

from pathlib import Path

from lightning.pytorch.callbacks import Checkpoint
from lightning.fabric.utilities.cloud_io import get_filesystem, _is_local_file_protocol
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

class MainCallback(Callback):
    def __init__(
        self, 
        dirpath,
        patience, 
        early_stopping_monitor='val_loss',
        early_stopping_mode="min",
        min_delta=0.0,
        early_stopping_verbose=True,
        log_rank_zero_only=True,
        save_top_k=5,
        checkpoint_monitor='val_f1_score',
        checkpoint_mode='max',
        verbose=True,
        early_stopping_active=True,
        model_checkpoint_active=True
    ):    
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        self.min_delta = min_delta
        self.patience = patience
        self.early_stopping_verbose = early_stopping_verbose
        self.log_rank_zero_only = log_rank_zero_only

        self.save_top_k = save_top_k
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.dirpath = dirpath
        self.verbose = verbose

        self.callback_list = [self.model_checkpoint(), self.early_stopping()]

    def early_stopping(self):
        if early_stopping_active:
            return EarlyStopping(
                monitor=self.early_stopping_monitor,
                mode=self.early_stopping_mode,
                # min_delta=self.min_delta,
                # patience=self.patience, 
                # verbose=self.verbose,
                # check_finite=True,
                # log_rank_zero_only=True
            )

    def model_checkpoint(self):
        if model_checkpoint_active:
            return ModelCheckpoint(
                save_top_k=self.save_top_k,
                monitor=self.checkpoint_monitor,
                mode=self.checkpoint_mode,
                # dirpath = model_folder,
                filename ='model-{val_loss:3f}-{val_f1_score:.3f}-{val_precision:.3f}-{val_recall:.3f}',
                save_on_train_epoch_end=False,
                verbose=self.verbose,
            )
    
    def setup(self):
        model_folder_name = f"{self.__class__.__name__}_{self.encoder_name}"
        # root_folder = Path(root_folder)
        # folder_name = f"{self.__class__.__name__}_{self.encoder_name}"
        # subfolder_name = f"input={self.input_size}_lr={self.lr:.2e}"
        # folder_path = root_folder / folder_name / subfolder_name
        # folder_path.mkdir(parents=True, exist_ok=True)
        # return folder_path

    def logger(self):
        return CSVLogger(save_dir=self.dirpath, name='model_logs')

    def __call__(self):
        return self.callback_list


class NewModelCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def setup(self, trainer, pl_module, stage):
        model_folder_name = Path(f"{pl_module.__class__.__name__}_{pl_module.encoder_name}")
        subfolder_name = f"input={pl_module.input_size}_lr={pl_module.lr:.2e}"
        self.dirpath = model_folder_name / subfolder_name
        self._fs = get_filesystem(self.dirpath or "")


class MetricsPlotter(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.metrics = {'loss': [], 'val_loss': [], 'train_f1_score': [], 'val_f1_score': []}

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self.max_epochs = trainer.max_epochs
    #     for key in self.metrics.keys():
    #         val = trainer.logged_metrics.get(key, 0)
    #         if isinstance(val, torch.Tensor):
    #             val = val.cpu().numpy().item()
    #         self.metrics[key].append(val)
    #     self.epoch += 1

    #     print(self.metrics)


if __name__ == '__main__':
    JSON_PATH_SPLIT = "/mnt/camobi_3/new_data/train_val.json"
    files_path = Path('/mnt/camobi_process/new_rioss_dataset')
    file_path = files_path / 'D205.nc'
    label_order_path = '/mnt/camobi_3/new_data/label_order.json'