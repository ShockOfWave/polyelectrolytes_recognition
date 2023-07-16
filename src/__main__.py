import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from tqdm import tqdm as status_loading
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import scikitplot as skplt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from src.data.datasets import *
from src.models.model import *

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

if __name__ == '__main__':
    
    writer = SummaryWriter('torchlogs/barcode_auto')
    
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images = np.load(os.path.join('data', 'full_data', 'barcode_images.npy'), allow_pickle=True)
    
    images_tmp = []
    
    for img in images:
        images_tmp.append(Image.fromarray(img).convert('RGB'))
        
    images = images_tmp
    
    table = pd.read_csv(os.path.join('data', 'full_data', 'auto_dataset.csv'))
    num_features_table = len(table.columns)
    y_data = np.load(os.path.join('data', 'full_data', 'y_data.npy'))
    
    le = LabelEncoder()
    std = StandardScaler()
    scaler = MinMaxScaler()
    y_data = le.fit_transform(y_data)
    table = std.fit_transform(table)
    table = scaler.fit_transform(table)
    
    pickle.dump(le, open('multimodal/barcode_auto/le.pkl', 'wb'))
    pickle.dump(std, open('multimodal/barcode_auto/std.pkl', 'wb'))
    pickle.dump(scaler, open('multimodal/barcode_auto/scaler.pkl', 'wb'))
    
    X_trainval_images, X_test_images, X_trainval_table, X_test_table, y_trainval, y_test = train_test_split(images, table, y_data, test_size=0.2, random_state=2)
    X_train_images, X_val_images, X_train_table, X_val_table, y_train, y_val = train_test_split(X_trainval_images, X_trainval_table, y_trainval, test_size=0.1, random_state=2)
    
    train_dataset = CustomCombinedDataset(X_train_images, X_train_table, y_train, transform=True)
    val_dataset = CustomCombinedDataset(X_val_images, X_val_table, y_val, transform=True)
    test_dataset = CustomCombinedDataset(X_test_images, X_test_table, y_test, transform=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    
    model = Net(num_features=num_features_table, num_class=3)
    # model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    
    num_epochs = 0
    
    for epoch in range(1, EPOCHS+1):
        
        train_epoch_loss = 0
        train_epoch_acc = 0
        num_epochs+=1
        
        # weight_histograms(writer, num_epochs, model)
        
        model.train()
        for imgs, tables, labels in status_loading(train_dataloader, desc='Training...'):
            imgs, tables, labels = Variable(imgs.to(device).float()), Variable(tables.to(device).float()), Variable(labels.to(device).long())
            if num_epochs == 1:
                writer.add_graph(model, input_to_model=(imgs, tables))
                writer.add_image('Example input', imgs[0])
            
            optimizer.zero_grad()
            
            y_train_pred = model(imgs, tables).squeeze()
            
            train_loss = criterion(y_train_pred, labels)
            train_acc = multi_acc(y_train_pred, labels)
            writer.add_scalar('Train/Loss', train_loss, num_epochs)
            writer.add_scalar('Train/Acc', train_acc, num_epochs)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            
            model.eval()
            for imgs, tables, labels in status_loading(val_dataloader, desc='Validation...'):
                imgs, tables, labels = Variable(imgs.to(device).float()), Variable(tables.to(device).float()), Variable(labels.to(device).long())
                y_val_pred = model(imgs, tables).squeeze()
                
                val_loss = criterion(y_val_pred, labels)
                val_acc = multi_acc(y_val_pred, labels)
                writer.add_scalar('Val/Loss', val_loss, num_epochs)
                writer.add_scalar('Val/Acc', val_acc, num_epochs)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        
        loss_stats['train'].append(train_epoch_loss/ len(train_dataloader))
        loss_stats['val'].append(val_epoch_loss/len(val_dataloader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_dataloader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_dataloader))
    
        print(f'Epoch {num_epochs}: | Train Loss: {train_epoch_loss/len(train_dataloader):.5f} | Val Loss: {val_epoch_loss/len(val_dataloader):.5f} | Train Acc: {train_epoch_acc/len(train_dataloader):.3f} | Val Acc: {val_epoch_acc/len(val_dataloader):.3f}')
    
    writer.flush()
    writer.close()
    
    y_pred_list = []
    y_probs_list = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for imgs, tables, labels in test_dataloader:
            imgs, tables = Variable(imgs.to(device).float()), Variable(tables.to(device).float())
            y_true.append(labels)
            y_test_pred = model(imgs, tables).squeeze()
            y_probs_list.append(y_test_pred.cpu().numpy())
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    pred_final = []
    probs_final = []
    y_true_final = []
    for preds in y_pred_list:
        for pred in preds:
            pred_final.append(pred)
            
    for probs in y_probs_list:
        for prob in probs:
            probs_final.append(prob)
            
    for y_tr in y_true:
        for y in y_tr:
            y_true_final.append(np.array(y))
    
    np.save('multimodal/barcode_auto/preds.npy', np.array(pred_final), allow_pickle=True)
    np.save('multimodal/barcode_auto/probs.npy', np.array(probs_final), allow_pickle=True)
    np.save('multimodal/barcode_auto/y_true.npy', np.array(y_true_final), allow_pickle=True)
    model.to('cpu')
    torch.save(model, 'multimodal/barcode_auto/model.pth')
    
    skplt.metrics.plot_confusion_matrix(y_true_final, pred_final, normalize=True)
    plt.savefig('multimodal/barcode_auto/cm.svg', format='svg', dpi=500)
    skplt.metrics.plot_roc(y_true_final, probs_final)
    plt.savefig('multimodal/barcode_auto/roc.svg', format='svg', dpi=500)
    skplt.metrics.plot_precision_recall(y_true_final, probs_final)
    plt.savefig('multimodal/barcode_auto/pre_recall.svg', format='svg', dpi=500)
    
    # print('Begin training.')
    # num_epochs = 0
    # for e in tqdm(range(1, EPOCHS+1)):
        
    #     # TRAINING
    #     train_epoch_loss = 0
    #     train_epoch_acc = 0
    #     num_epochs+=1

    #     model.train()
    #     for X_train_batch, y_train_batch in train_loader:
    #         X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
    #         optimizer.zero_grad()
            
    #         y_train_pred = model(X_train_batch).squeeze()
            
    #         train_loss = criterion(y_train_pred, y_train_batch)
    #         train_acc = multi_acc(y_train_pred, y_train_batch)
            
    #         train_loss.backward()
    #         optimizer.step()
            
    #         train_epoch_loss += train_loss.item()
    #         train_epoch_acc += train_acc.item()
            
    #     with torch.no_grad():
    #         val_epochs_loss = 0
    #         val_epochs_acc = 0
            
    #         model.eval()
    #         for X_val_batch, y_val_batch in val_loader:
    #             X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
    #             y_val_pred = model(X_val_batch).squeeze()
                
    #             val_loss = criterion(y_val_pred, y_val_batch)
    #             val_acc = multi_acc(y_val_pred, y_val_batch)
                
    #             val_epochs_loss += val_loss.item()
    #             val_epochs_acc += val_acc.item()
        
    #     loss_stats['train'].append(train_epoch_loss/ len(train_loader))
    #     loss_stats['val'].append(val_epochs_loss/len(val_loader))
    #     accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    #     accuracy_stats['val'].append(val_epochs_acc/len(val_loader))
        
        # print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epochs_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f} | Val Acc: {val_epochs_acc/len(val_loader):.3f}')