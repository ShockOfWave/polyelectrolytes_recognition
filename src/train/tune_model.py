import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter

from src.paths.paths import get_project_path
from src.models.model import Combined_Model
from src.data.data import CustomCombinedDataset
from src.utils.config import tune_config

def train_tune(config, num_epochs=10, num_gpus=0):
    
    model = Combined_Model(config)
    
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=20)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices=math.ceil(num_gpus),
        logger = TensorBoardLogger(
            save_dir=os.path.join(get_project_path(), 'logs'), name=''
        ),
        enable_progress_bar=False,
        log_every_n_steps=1,
        callbacks=[
            TuneReportCallback(
                {
                    'val_loss': 'val_loss_epoch',
                    'val_accuracy': 'val_acc_epoch'
                },
                on='validation_end'
            )
        ]
    )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
def tune_model(num_samples=10, num_epochs=10, gpus_per_trial=0):
    
    config = tune_config
    
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        parameter_columns=['sign_size', 'cha_input', 'cha_hidden', 'K', 'dropout_input', 'dropout_hidden', 'dropout_output', 'N', 'hidden_layer', 'dropout_size'],
        metric_columns=['val_loss', 'val_accuracy', 'training_iteration']
    )
    
    train_fn_with_parameters = tune.with_parameters(train_tune,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial)
    
    resources_per_trial = {'cpu': 20, 'gpu': gpus_per_trial}
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            storage_path=os.path.join(get_project_path(), 'ray_results'),
            name='tune_AFM_TDA',
            progress_reporter=reporter
        ),
        param_space=config
    )
    
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)

    

if __name__ == "__main__":
    tune_model(gpus_per_trial=1, num_samples=50, num_epochs=10)