import numpy as np
import os

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from src.paths.paths import get_project_path
from src.data.data import CustomCombinedDataset
from src.models.model import Combined_Model
from src.utils.config import default_config

def train_model(config: dict = default_config):
    
    path_to_data = os.path.join(get_project_path(), 'prepared_data')
    
    auto = np.load(os.path.join(path_to_data, 'auto_tables.npy'), allow_pickle=True)
    min_max = np.load(os.path.join(path_to_data, 'min_max_tables.npy'), allow_pickle=True)
    flattened = np.load(os.path.join(path_to_data, 'flattened_tables.npy'), allow_pickle=True)

    barcode = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    diagram = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    afm = np.load(os.path.join(path_to_data, 'afm_images.npy'), allow_pickle=True)

    labels = np.load(os.path.join(path_to_data, 'labels.npy'), allow_pickle=True)

    train_auto, val_auto, train_min_max, val_min_max, train_flattened, val_flattened, train_barcode, val_barcode, train_diagram, val_diagram, train_afm, val_afm, y_train, y_val = train_test_split(auto, min_max, flattened, barcode, diagram, afm, labels, test_size=0.25, random_state=42)

    train_dataset = CustomCombinedDataset(train_auto, train_min_max, train_flattened, train_barcode, train_diagram, train_afm, y_train)
    val_dataset = CustomCombinedDataset(val_auto, val_min_max, val_flattened, val_barcode, val_diagram, val_afm, y_val)
    
    early_stop_callback = EarlyStopping(
                    monitor='val_loss_epoch',
                    min_delta=.01,
                    patience=20,
                    verbose=True,
                    mode='min'
                )
    
    logger = TensorBoardLogger(save_dir=os.path.join(get_project_path(), 'lightning_logs_2'))
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model = Combined_Model(config)
    
    trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=1,
                    min_epochs=5,
                    max_epochs=100,
                    log_every_n_steps=15,
                    callbacks=[checkpoint_callback, lr_monitor, early_stop_callback, TQDMProgressBar(refresh_rate=1)],
                    logger=logger
                )
    
    trainer.fit(model, DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4), DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4))
    
if __name__ == "__main__":
    train_model()