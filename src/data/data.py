import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pytorch_lightning as pl

from src.paths.paths import get_project_path

class CustomCombinedDataset(Dataset):
    def __init__(self, auto, min_max, flattened, barcode, diagram, afm, labels):
        # Tables
        self.auto = torch.tensor(auto, dtype=torch.float32)
        self.min_max = torch.tensor(min_max, dtype=torch.float32)
        self.flattened = torch.tensor(flattened, dtype=torch.float32)
        # Images
        self.barcode = barcode
        self.diagram = diagram
        self.afm = afm
        # Classes
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(labels)
        self.labels = torch.tensor(self.labels)
        # Define the transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(30),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        pickle.dump(self.le, open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le.pkl'), 'wb'))
    
    def __getitem__(self, index):
        # Get table by index
        auto = self.auto[index]
        min_max = self.min_max[index]
        flattened = self.flattened[index]
        # Get image by index
        barcode = self.transform(self.barcode[index])
        diagram = self.transform(self.diagram[index])
        afm = self.transform(self.afm[index])
        # Get label by index
        label = self.labels[index]
        return {'auto': auto, 'min_max': min_max, 'flattened': flattened, 'barcode': barcode, 'diagram': diagram, 'afm': afm, 'label': label}
    
    def __len__(self):
        return len(self.labels.numpy())
    
    def classes_(self):
        return np.unique(self.labels.numpy())
    
class CustomDataset_lightning(pl.LightningDataModule):
    
    def __init__(self, batch_size=2):
        super().__init__()
        
        self.path_to_data = os.path.join(get_project_path(), 'prepared_data')
        self.batch_size = batch_size
    
    def setup(self, stage):
        auto = np.load(os.path.join(self.path_to_data, 'auto_tables.npy'), allow_pickle=True)
        min_max = np.load(os.path.join(self.path_to_data, 'min_max_tables.npy'), allow_pickle=True)
        flattened = np.load(os.path.join(self.path_to_data, 'flattened_tables.npy'), allow_pickle=True)

        barcode = np.load(os.path.join(self.path_to_data, 'barcode_images.npy'), allow_pickle=True)
        diagram = np.load(os.path.join(self.path_to_data, 'barcode_images.npy'), allow_pickle=True)
        afm = np.load(os.path.join(self.path_to_data, 'afm_images.npy'), allow_pickle=True)

        labels = np.load(os.path.join(self.path_to_data, 'labels.npy'), allow_pickle=True)
        
        train_auto, val_auto, train_min_max, val_min_max, train_flattened, val_flattened, train_barcode, val_barcode, train_diagram, val_diagram, train_afm, val_afm, y_train, y_val = train_test_split(auto, min_max, flattened, barcode, diagram, afm, labels, test_size=0.25, random_state=42)

        self.train_dataset = CustomCombinedDataset(train_auto, train_min_max, train_flattened, train_barcode, train_diagram, train_afm, y_train)
        self.val_dataset = CustomCombinedDataset(val_auto, val_min_max, val_flattened, val_barcode, val_diagram, val_afm, y_val)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)