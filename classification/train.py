# pytorch training
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchaudio 

import timm

# general
import argparse
import librosa
import os
import numpy as np

# logging
import wandb
import datetime
time_now  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 

# other files 
from dataset import BirdCLEFDataset, get_datasets
from model import BirdCLEFModel, GeM

# cmap metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_print_freq = 50 

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=264, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=16, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=16, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)


#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
    
def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()

    running_loss = 0
    log_n = 0
    log_loss = 0
    correct = 0
    total = 0

    for i, (mels, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        log_loss += loss.item()
        log_n += 1

        if i % (loss_print_freq) == 0 or i == len(data_loader) - 1:
            print("Loss:", log_loss, "Accuracy:", correct / total * 100.)
            log_loss = 0
            log_n = 0
            correct = 0
            total = 0

    return running_loss/len(data_loader)

def valid(model, data_loader, device, epoch):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    for i, (mels, labels) in enumerate(data_loader):
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_fn(outputs, labels)
            
        running_loss += loss.item()
        
        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())
    
    try:
        np.savetxt(f"{time_now}_{epoch}_labels.txt", label, delimiter=",")
        np.savetxt(f"{time_now}_{epoch}_predictions.txt", label, delimiter=",")
    except:
        print("L your txt(s) died") 
    
    # calculate MAP
    valid_map = mAP(label, pred)
    valid_f1 = f1_score(label, pred, average='macro')
    
    return running_loss/len(data_loader), valid_map, valid_f1

def mAP(label, pred):
    # one hot encoding
    y_label = label_binarize(label, classes=range(len(target_names)))
    y_pred = label_binarize(pred, classes=range(len(target_names)))
    
    # tp/fp/precision
    true_pos = ((y_label == 1) & (y_pred == 1)).sum(axis=0)
    false_pos = ((y_label == 0) & (y_pred == 1)).sum(axis=0)
    precision = true_pos / (true_pos + false_pos)
    precision = np.nan_to_num(precision)
    num_species = precision.shape[0]
    return precision.sum() / num_species

def set_seed():
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

if __name__ == '__main__':
    CONFIG = parser.parse_args()
    set_seed()
    print("Loading Model...")
    model = BirdCLEFModel(CONFIG=CONFIG).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Succesful :P")

    print("Loading Dataset")
    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs
    )
    
    print("Training")
    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        train_loss = train(
            model, 
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch)
        
        torch.save(model.state_dict(), f'./{time_now}_model_{epoch}.bin')
        
        valid_loss, valid_map, valid_f1 = valid(model, val_dataloader, device, epoch)
        print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map} \n Validation F1: \t{valid_f1}" )
        if valid_map > best_valid_map:
            print(f"Validation MAP Improved - {best_valid_map} ---> {valid_map}")
            torch.save(model.state_dict(), f'./{time_now}_model_{epoch}.bin')
            print(f"Saved model checkpoint at ./{time_now}_model_{epoch}.bin")
            best_valid_map = valid_map

    print(":o wow")