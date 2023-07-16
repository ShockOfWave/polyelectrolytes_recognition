import os
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray.tune.schedulers import ASHAScheduler

from src.paths.paths import get_project_path
from src.models.model import Combined_Model
from src.data.data import CustomDataset_lightning
from src.utils.config import tune_config

num_epochs = 10
num_samples = 500
accelerator = 'gpu'

if __name__ == "__main__":
    logger = TensorBoardLogger(save_dir=os.getcwd())
    
    path_to_data = os.path.join(get_project_path(), 'prepared_data')
    
    auto = np.load(os.path.join(path_to_data, 'auto_tables.npy'), allow_pickle=True)
    min_max = np.load(os.path.join(path_to_data, 'min_max_tables.npy'), allow_pickle=True)
    flattened = np.load(os.path.join(path_to_data, 'flattened_tables.npy'), allow_pickle=True)

    barcode = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    diagram = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    afm = np.load(os.path.join(path_to_data, 'afm_images.npy'), allow_pickle=True)

    labels = np.load(os.path.join(path_to_data, 'labels.npy'), allow_pickle=True)
    
    dm = CustomDataset_lightning()
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=Combined_Model, config=tune_config)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger)
        .fit_params(datamodule=dm)
        .checkpointing(monitor='val_acc_epoch', save_top_k=2, mode='max')
        .build()
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute='val_acc_epoch',
            checkpoint_score_order='max'
        )
    )
    
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"GPU": 2, 'CPU':40}
    )
    lightning_trainer = LightningTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
        
        tuner = tune.Tuner(
            lightning_trainer,
            param_space={"lightning_config": lightning_config},
            tune_config=tune.TuneConfig(
                metric='val_acc_epoch',
                mode='max',
                num_samples=num_samples,
                scheduler=scheduler
            ),
            run_config=air.RunConfig(
                name='tune_TDA_AFM',
                storage_path='./ray_results'
            ),
        )
        
        results = tuner.fit()
        best_results = results.get_best_result(metric='val_acc_epoch', mode='max')
        print("Best hyperparameter found were: ", results.get_best_result().config)
        results.get_dataframe().to_csv('results_table.csv')
    
    tune_asha(num_samples=num_samples)