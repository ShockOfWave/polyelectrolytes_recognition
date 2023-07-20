import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv, find_dotenv

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from torch.utils.data import DataLoader

from lightning.pytorch.loggers.neptune import NeptuneLogger

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary

from src.models.model import Combined_Model
from src.data.data import CustomCombinedDataset
from src.paths.paths import get_project_path

def objective(trial: optuna.trial.Trial):
    
    tune_config = {
        'sign_size': trial.suggest_int('sign_size', 32, 32, step=16),
        'cha_input': trial.suggest_int('cha_input', 16, 16, step=16),
        'cha_hidden': trial.suggest_int('cha_hidden', 256, 1024, step=256),
        'K': trial.suggest_int('K', 3, 7, step=2),
        'dropout_input': trial.suggest_float('dropout_input', 0.1, 0.1),
        'dropout_hidden': trial.suggest_float('dropout_hidden', 0.1, 0.1),
        'dropout_output': trial.suggest_float('dropout_output', 0.1, 0.1),
        'N': trial.suggest_int('N', 3, 7, step=2),
        'hidden_layer': trial.suggest_int('hidden_layer', 256, 1024, step=256),
        'dropout_size': trial.suggest_float('dropout_size', 0.05, 0.05)
    }
    
    model = Combined_Model(tune_config)
    
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
    test_dataset = CustomCombinedDataset(np.array(auto), np.array(min_max), np.array(flattened), np.array(barcode), np.array(diagram), np.array(afm), np.array(labels))
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    callback = PyTorchLightningPruningCallback(trial, monitor='val_acc_epoch')
    
    neptune_logger = NeptuneLogger(
        api_key=os.environ.get('api_token_neptune'),
        project='shockofwave/polyelectrolytes-recognition',
        tags=['training', 'multihead'],
        mode='offline'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss_epoch',
        min_delta=0.01,
        mode='min',
        patience=20
    )
    
    rich_progress_bar = RichProgressBar()
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    rish_model_summary = RichModelSummary()
    
    trainer = Trainer(
        logger=neptune_logger,
        enable_checkpointing=False,
        max_epochs=100,
        accelerator='auto',
        devices=[0],
        callbacks=[callback, early_stopping, rich_progress_bar, lr_monitor, rish_model_summary],
        log_every_n_steps=1
    )
    
    trainer.logger.log_hyperparams(tune_config)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    
    return trainer.callback_metrics['val_loss_epoch'].item()

def tune_model_optuna():
    load_dotenv(find_dotenv())
    
    study = optuna.create_study(
        study_name='multihead_nn_tda_afm_analysis',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
        storage=f'sqlite:///{get_project_path()}/optuna_models_optimization.db'
    )
    
    study.optimize(objective, n_trials=100)
    
    joblib.dump(study, os.path.join(get_project_path(), 'study.pkl'))
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    tune_model_optuna()
        
    
