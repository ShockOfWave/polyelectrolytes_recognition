import os
import io
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from sklearn.metrics import *
import scikitplot as skplt

from src.paths.paths import get_project_path

class Table_NN(nn.Module):
    def __init__(self, input_dim, sign_size=32, cha_input=16, cha_hidden=1024,
                 K=3, dropout_input=0.1, dropout_hidden=0.1, dropout_output=0.1):
        super().__init__()
        
        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride = 1, 
            padding=2,  
            groups=cha_input, 
            bias=True)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=True)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=True)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=True)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        return x
    
class Image_NN(nn.Module):
    def __init__(self, input_size=(256, 256, 3), K=7, hidden_layer=512, dropout_size=0.05):
        super().__init__()
        
        self.in_channels = input_size[-1]
        self.K = K
        self.hidden_layer = hidden_layer
        self.dropout_size = dropout_size
        self.output_size = self.hidden_layer//(K*7)
        
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_layer//4, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Dropout2d(self.dropout_size)
        )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_layer//4),
            nn.Conv2d(in_channels=self.hidden_layer//4, out_channels=self.hidden_layer//2, kernel_size=7, stride=1, padding=9, dilation=3)
        )
        
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_layer//2),
            nn.Conv2d(in_channels=self.hidden_layer//2, out_channels=self.hidden_layer, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.dropout_size)
        )
        
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_layer),
            nn.Conv2d(in_channels=self.hidden_layer, out_channels=self.hidden_layer//K, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_layer//K),
            nn.Conv2d(in_channels=self.hidden_layer//K, out_channels=self.hidden_layer//(K*3), kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(self.dropout_size)
        )
        
        self.conv6 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_layer//(K*3)),
            nn.Conv2d(in_channels=self.hidden_layer//(K*3), out_channels=self.hidden_layer//(K*7), kernel_size=3, padding=2, dilation=2),
            nn.MaxPool2d(2, 2),
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        
        return x
    
default_config = {
    'sign_size': 32,
    'cha_input': 16,
    'cha_hidden': 1024,
    'K': 3,
    'dropout_input': 0.1,
    'dropout_hidden': 0.1,
    'dropout_output': 0.1,
    'N': 7,
    'hidden_layer': 512,
    'dropout_size': 0.05
}
 
class Combined_Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # create default parameters
        self.input_dim_auto = 1536
        self.input_dim_min_max = 36
        self.input_dim_flattened = 50575
        self.input_size = (256, 256, 3)
        self.output_dim = 5
        self.le = pickle.load(open(os.path.join(get_project_path(), 'models', 'preprocessing', 'le.pkl'), 'rb'))
        
        # parse config dict
        self.sign_size = config['sign_size']
        self.cha_input = config['cha_input']
        self.cha_hidden = config['cha_hidden']
        self.K = config['K']
        self.dropout_input = config['dropout_input']
        self.dropout_hidden = config['dropout_hidden']
        self.dropout_output = config['dropout_output']
        self.N = config['N']
        self.hidden_layer = config['hidden_layer']
        self.dropout_size = config['dropout_size']
        
        self.save_hyperparameters()
        self.output_size_images = (self.hidden_layer//(self.N*7))*16*16
        self.output_size_table = (self.sign_size//4) * self.cha_hidden
        
        # lists for outputs
        # train
        self.train_losses_all: list = []
        self.train_probs_all: list = []
        self.train_y_all: list = []
        # validation
        self.val_losses_all: list = []
        self.val_probs_all: list = []
        self.val_y_all: list = []
        
        
        self.inn1 = Image_NN(
            self.input_size,
            self.N,
            self.hidden_layer,
            self.dropout_size
        )
        
        self.inn2 = Image_NN(
            self.input_size,
            self.N,
            self.hidden_layer,
            self.dropout_size
        )
        
        self.inn3 = Image_NN(
            self.input_size,
            self.N,
            self.hidden_layer,
            self.dropout_size
        )
        
        self.tnn1 = Table_NN(
            self.input_dim_auto,
            self.sign_size,
            self.cha_input,
            self.cha_hidden,
            self.K,
            self.dropout_input,
            self.dropout_hidden,
            self.dropout_output
        )
        
        self.tnn2 = Table_NN(
            self.input_dim_min_max,
            self.sign_size,
            self.cha_input,
            self.cha_hidden,
            self.K,
            self.dropout_input,
            self.dropout_hidden,
            self.dropout_output
        )
        
        self.tnn3 = Table_NN(
            self.input_dim_flattened,
            self.sign_size,
            self.cha_input,
            self.cha_hidden,
            self.K,
            self.dropout_input,
            self.dropout_hidden,
            self.dropout_output
        )
        
        fc_inn1 = nn.Linear(self.output_size_images, self.output_dim)
        self.fc_inn1 = nn.utils.weight_norm(fc_inn1)
        
        fc_inn2 = nn.Linear(self.output_size_images, self.output_dim)
        self.fc_inn2 = nn.utils.weight_norm(fc_inn2)
        
        fc_inn3 = nn.Linear(self.output_size_images, self.output_dim)
        self.fc_inn3 = nn.utils.weight_norm(fc_inn3)
        
        fc_tnn1 = nn.Linear(self.output_size_table, self.output_dim)
        self.fc_tnn1 = nn.utils.weight_norm(fc_tnn1)
        
        fc_tnn2 = nn.Linear(self.output_size_table, self.output_dim)
        self.fc_tnn2 = nn.utils.weight_norm(fc_tnn2)
        
        fc_tnn3 = nn.Linear(self.output_size_table, self.output_dim)
        self.fc_tnn3 = nn.utils.weight_norm(fc_tnn3)
        
        self.bn = nn.BatchNorm1d(self.output_dim)
        fc = nn.Linear(self.output_dim, self.output_dim)
        self.fc = nn.utils.weight_norm(fc)
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, auto, min_max, flattened, barcode, diagram, afm):
        
        x1 = self.tnn1(auto)
        x1 = self.fc_tnn1(x1)
        
        x2 = self.tnn2(min_max)
        x2 = self.fc_tnn2(x2)
        
        x3 = self.tnn3(flattened)
        x3 = self.fc_tnn3(x3)
        
        x4 = self.inn1(barcode)
        x4 = self.fc_inn1(x4)
        
        x5 = self.inn2(diagram)
        x5 = self.fc_inn2(x5)
        
        x6 = self.inn3(afm)
        x6 = self.fc_inn3(x6)
        
        x = x1 + x2 + x3 + x4 + x5 + x6
        
        return x
    
    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, x6, y = batch['auto'], batch['min_max'], batch['flattened'], batch['barcode'], batch['diagram'], batch['afm'], batch['label']
        y_hat = self.forward(x1, x2, x3, x4, x5, x6)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=False)
        
        self.train_losses_all.append(loss)
        self.train_probs_all.append(y_hat)
        self.train_y_all.append(y)
                
    
    def on_training_epoch_end(self):
        
        if self.current_epoch == 1:
            new_batch = {
                'auto': torch.rand((1, 1536)),
                'min_max': torch.rand((1, 36)),
                'flattened': torch.rand((1, 50575)),
                'barcode': torch.rand((1, 3, 256, 256)),
                'diagram': torch.rand((1, 3, 256, 256)),
                'afm': torch.rand((1, 3, 256, 256))
            }
            self.logger.experiment.add_graph(self, (new_batch['auto'], new_batch['min_max'], new_batch['flattened'], new_batch['barcode'], new_batch['diagram'], new_batch['afm']))
        
        avg_loss = torch.stack(self.train_losses_all).mean()
        y_probs = torch.cat(self.train_probs_all, 0)
        y_true = torch.cat(self.train_y_all, 0)
        
        acc = accuracy_score(y_true.cpu(), y_probs.argmax(1).cpu().numpy())
        
        self.log('train_loss_epoch', avg_loss, sync_dist=True, prog_bar=True)
        self.log('train_acc_epoch', acc, sync_dist=True, prog_bar=True)
        
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Train', acc, self.current_epoch)
        
        if not self.le == None:
            skplt.metrics.plot_confusion_matrix(self.le.inverse_transform(y_true.cpu().numpy()), self.le.inverse_transform(y_probs.argmax(1).cpu().numpy()), normalize=True)
        else:
            skplt.metrics.plot_confusion_matrix(y_true.cpu().numpy(), y_probs.argmax(1).cpu().numpy(), normalize=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=500)
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        
        self.logger.experiment.add_image('Training confusion matrix', im, self.current_epoch)
        
        self.train_losses_all.clear()
        self.train_probs_all.clear()
        self.train_y_all.clear()

        
    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, x6, y = batch['auto'], batch['min_max'], batch['flattened'], batch['barcode'], batch['diagram'], batch['afm'], batch['label']
        y_hat = self.forward(x1, x2, x3, x4, x5, x6)
        loss = self.loss(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=False)
        
        self.val_losses_all.append(loss)
        self.val_probs_all.append(y_hat)
        self.val_y_all.append(y)
        

    def on_validation_epoch_end(self):
        
        avg_loss = torch.stack(self.val_losses_all).mean()
        y_probs = torch.cat(self.val_probs_all, 0)
        y_true = torch.cat(self.val_y_all, 0)
        
        acc = accuracy_score(y_true.cpu(), y_probs.argmax(1).cpu().numpy())
        
        self.log('val_loss_epoch', avg_loss, sync_dist=True, prog_bar=True)
        self.log('val_acc_epoch', acc, sync_dist=True, prog_bar=True)
        
        self.logger.experiment.add_scalar('Loss/Valid', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Valid', acc, self.current_epoch)
        if not self.le == None:
            skplt.metrics.plot_confusion_matrix(self.le.inverse_transform(y_true.cpu().numpy()), self.le.inverse_transform(y_probs.argmax(1).cpu().numpy()), normalize=True)
        else:
            skplt.metrics.plot_confusion_matrix(y_true.cpu().numpy(), y_probs.argmax(1).cpu().numpy(), normalize=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=500)
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        
        self.logger.experiment.add_image('Validation confusion matrix', im, self.current_epoch)
        
        self.val_losses_all.clear()
        self.val_probs_all.clear()
        self.val_y_all.clear()
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-8),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]
    
    # def setup(self):
    #     path_to_data = os.path.join(get_project_path(), 'prepared_data')
    
    #     auto = np.load(os.path.join(path_to_data, 'auto_tables.npy'), allow_pickle=True)
    #     min_max = np.load(os.path.join(path_to_data, 'min_max_tables.npy'), allow_pickle=True)
    #     flattened = np.load(os.path.join(path_to_data, 'flattened_tables.npy'), allow_pickle=True)

    #     barcode = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    #     diagram = np.load(os.path.join(path_to_data, 'barcode_images.npy'), allow_pickle=True)
    #     afm = np.load(os.path.join(path_to_data, 'afm_images.npy'), allow_pickle=True)

    #     labels = np.load(os.path.join(path_to_data, 'labels.npy'), allow_pickle=True)

    #     train_auto, val_auto, train_min_max, val_min_max, train_flattened, val_flattened, train_barcode, val_barcode, train_diagram, val_diagram, train_afm, val_afm, y_train, y_val = train_test_split(auto, min_max, flattened, barcode, diagram, afm, labels, test_size=0.25, random_state=42)

    #     self.train_dataset = CustomCombinedDataset(train_auto, train_min_max, train_flattened, train_barcode, train_diagram, train_afm, y_train)
    #     self.val_dataset = CustomCombinedDataset(val_auto, val_min_max, val_flattened, val_barcode, val_diagram, val_afm, y_val)
        
    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return DataLoader(self.train_dataset, batch_size=4)
    
    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.val_dataset, batch_size=4)