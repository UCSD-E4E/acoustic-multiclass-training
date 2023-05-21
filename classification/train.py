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
from tqdm import tqdm
# # cmap metrics
# import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import MultilabelAveragePrecision
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=574, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=16, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=16, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)
parser.add_argument('-l', '--logging', default='True', type=str)
parser.add_argument('-lf', '--logging_freq', default=20, type=int)
parser.add_argument('-vf', '--valid_freq', default=2000, type=int)
parser.add_argument('-mch', '--model_checkpoint', default=None, type=str)
parser.add_argument('-md', '--map_debug', action='store_true')
parser.add_argument('-p', '--p', default=0, type=float, help='p for mixup')
parser.add_argument('-i', '--imb', action='store_true', help='imbalance sampler')
parser.add_argument('-pw', "--pos_weight", type=float, default=1, help='pos weight')
parser.add_argument('-lr', "--lr", type=float, default=1e-3, help='learning rate')

#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training



    
def train(model, data_loader, optimizer, scheduler, device, step, best_valid_cmap, epoch):
    print('size of data loader:', len(data_loader))
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
        # sigmoid multilabel predictions
        preds = torch.sigmoid(outputs) > 0.5
        
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        
        if scheduler is not None:
            scheduler.step()

        if np.isnan(loss.item()):
            print(mels)
            print(labels)
            print(outputs)
            raise "NAN ERROR"   
        running_loss += loss.item()
        total += labels.size(0)
        correct += torch.all(preds.eq(labels), dim=-1).sum().item()
        log_loss += loss.item()
        log_n += 1

        if i % (CONFIG.logging_freq) == 0 or i == len(data_loader) - 1:
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/accuracy": correct / total * 100.,
                "custom_step": step,
            })
            print("Loss:", log_loss / log_n, "Accuracy:", correct / total * 100.)
            log_loss = 0
            log_n = 0
            correct = 0
            total = 0
        
        if step % CONFIG.valid_freq == 0 and step != 0:
            del mels, labels, outputs, preds # clear memory
            valid_loss, valid_map = valid(model, val_dataloader, device, step)
            print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map}" )
            if valid_map > best_valid_cmap:
                print(f"Validation cmAP Improved - {best_valid_cmap} ---> {valid_map}")
                best_valid_cmap = valid_map
                torch.save(model.state_dict(), run.name + '.pt')
                print(run.name + '.pt')
            model.train()
            
        
        step += 1

    return running_loss/len(data_loader), step, best_valid_cmap

def valid(model, data_loader, device, step, pad_n=5):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    dl = tqdm(data_loader, position=5)
    if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
        pred = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
        label = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')
    else:
        for i, (mels, labels) in enumerate(dl):
            mels = mels.to(device)
            labels = labels.to(device)
            
            # argmax
            outputs = model(mels)
            _, preds = torch.max(outputs, 1)
            
            loss = loss_fn(outputs, labels)
                
            running_loss += loss.item()
            
            pred.append(outputs.cpu().detach())
            label.append(labels.cpu().detach())
            # break
        pred = torch.cat(pred)
        label = torch.cat(label)
        if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
            torch.save(pred, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
            torch.save(label, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')

    # print(torch.unique(label))
    # convert to one-hot encoding
    unq_classes = torch.unique(label)
    print(unq_classes)
    if label.shape[1] < CONFIG.num_classes:
        label = F.one_hot(label, num_classes=CONFIG.num_classes).to(device)
    # label = label[:,unq_classes]

    # softmax predictions
    pred = F.sigmoid(pred).to(device)
    # pred = pred[:, unq_classes]

    # # pad predictions and labels with `pad_n` true positives
    padded_preds = torch.cat([pred, torch.ones(pad_n, pred.shape[1]).to(pred.device)])
    padded_labels = torch.cat([label, torch.ones(pad_n, label.shape[1]).to(label.device)])
    print(label.shape, pred.shape) 
    # print(padded_labels.shape, padded_preds.shape)
    # send to cpu
    padded_preds = padded_preds.detach().cpu()
    padded_labels = padded_labels.detach().cpu().long()
    # padded_preds = padded_preds.detach().cpu()
    # padded_labels = padded_labels.detach().cpu()

    metric = MultilabelAveragePrecision(num_labels=CONFIG.num_classes, average="weighted")
    valid_map = metric(padded_preds, padded_labels)
    # calculate average precision
    # valid_map = average_precision_score(
    #     padded_labels,
    #     padded_preds,
    #     average='macro',
    # )
    # _, padded_preds = torch.max(padded_preds, 1)

    # acc = (padded_preds == padded_labels).sum().item() / len(padded_preds)
    # print("Validation Accuracy:", acc)
    
    
    print("Validation mAP:", valid_map)
    
    wandb.log({
        "valid/loss": running_loss/len(data_loader),
        "valid/cmap": valid_map,
        "custom_step": step,
        
    })
    
    return running_loss/len(data_loader), valid_map

def set_seed():
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)


def init_wandb(CONFIG):
    run = wandb.init(
        project="birdclef-2023",
        config=CONFIG,
        mode="disabled" if CONFIG.logging == False else "online"
    )
    run.name = f"EFN-{CONFIG.epochs}-{CONFIG.train_batch_size}-{CONFIG.valid_batch_size}-{CONFIG.sample_rate}-{CONFIG.hop_length}-{CONFIG.max_time}-{CONFIG.n_mels}-{CONFIG.n_fft}-{CONFIG.seed}-" + run.name.split('-')[-1]
    return run

if __name__ == '__main__':
    print(device)
    CONFIG = parser.parse_args()
    print(CONFIG)
    CONFIG.logging = True if CONFIG.logging == 'True' else False
    run = init_wandb(CONFIG)
    set_seed()
    print("Loading Model...")
    model = BirdCLEFModel(CONFIG=CONFIG).to(device)
    if CONFIG.model_checkpoint is not None:
        model.load_state_dict(torch.load(CONFIG.model_checkpoint))
    optimizer = Adam(model.parameters(), lr=CONFIG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Succesful :P")

    print("Loading Dataset")
    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs,
        collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs,
        collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    )
    
    print("Training")
    step = 0
    best_valid_cmap = 0

    if not CONFIG.imb: # normal loss
        if CONFIG.pos_weight != 1:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG.pos_weight] * CONFIG.num_classes).to(device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    else: # weighted loss
        if CONFIG.pos_weight != 1:
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([CONFIG.pos_weight] * CONFIG.num_classes).to(device),
                weight=torch.tensor([1 / p for p in train_dataset.class_id_to_num_samples.values()]).to(device)
            )
        else:
            loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1 / p for p in train_dataset.class_id_to_num_samples.values()]).to(device))
    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        train_loss, step, best_valid_cmap = train(
            model, 
            train_dataloader,
            optimizer,
            scheduler,
            device,
            step,
            best_valid_cmap,
            epoch
        )
        valid_loss, valid_map = valid(model, val_dataloader, device, step)
        print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map}" )

        if valid_map > best_valid_cmap:
            torch.save(model.state_dict(), run.name + '.pt')
            print(run.name + '.pt')
            print(f"Validation cmAP Improved - {best_valid_cmap} ---> {valid_map}")
            best_valid_cmap = valid_map
        


    print(":o wow")