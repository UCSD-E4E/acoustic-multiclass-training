"""
    Contains the training and validation function and logging to Weights and Biases
    Methods:
        train: trains the model
        valid: calculates validation loss and accuracy
        set_seed: sets the random seed
        init_wandb: initializes the Weights and Biases logging
        

"""

# other files 
from dataset import PyhaDF_Dataset, get_datasets
from model import BirdCLEFModel, GeM
from tqdm import tqdm
from utils import set_seed, print_verbose
from config import get_config

# pytorch training
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim import Adam

# general
import numpy as np
from typing import Dict, Any, Tuple
import os



# other files 
from model import BirdCLEFModel #, GeM
from tqdm import tqdm
tqdm.pandas()
from torchmetrics.classification import MultilabelAveragePrecision


# pytorch training
import torch
import torch.nn as nn


#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training

# logging
import wandb
import datetime
time_now  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 

from torchmetrics.classification import MultilabelAveragePrecision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def train(model: BirdCLEFModel,
        data_loader: PyhaDF_Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        step: int,
        best_valid_cmap: float,
        epoch: int,
        CONFIG) -> Tuple[float, int, float]:
    """ Trains the model
        Returns: 
            loss: the average loss over the epoch
            step: the current step
            best_valid_cmap: the best validation mAP
    """
    print_verbose('size of data loader:', len(data_loader),verbose=CONFIG.verbose)
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
        # preds = torch.sigmoid(outputs) > 0.5
        

        loss = model.loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        total += labels.size(0)

        # index of highest predicted class
        #pred_label = torch.argmax(outputs, dim=1)

        # checking highest against true label

        correct += torch.all(torch.round(outputs).eq(labels), dim=-1).sum().item()
        log_loss += loss.item()
        log_n += 1

        if i % (CONFIG.logging_freq) == 0 or i == len(data_loader) - 1:
            #Log to Weights and Biases
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
        
        #if step % CONFIG.valid_freq == 0 and step != 0:
        #    del mels, labels, outputs, preds # clear memory
        #    valid_loss, valid_map = valid(model, val_dataloader, device, step)
        #    print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map}" )
        #    if valid_map > best_valid_cmap:
        #        print(f"Validation cmAP Improved - {best_valid_cmap} ---> {valid_map}")
        #        best_valid_cmap = valid_map
        #        torch.save(model.state_dict(), wandb_run.name + '.pt')
        #        print(wandb_run.name + '.pt')
        #    model.train()
            
        
        step += 1

    return running_loss/len(data_loader), step, best_valid_cmap

def valid(model: BirdCLEFModel,
          data_loader: PyhaDF_Dataset,
          device: str,
          step: int,
          CONFIG) -> Tuple[float, float]:
    """
    Run a validation loop
    """
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    # tqdm is a progress bar
    dl = tqdm(data_loader, position=5)
    
    if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
        pred = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
        label = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')
    else:
        for _, (mels, labels) in enumerate(dl):
            mels = mels.to(device)
            labels = labels.to(device)
            
            # argmax
            outputs = model(mels)
            #_, preds = torch.max(outputs, 1)
            
            loss = model.loss_fn(outputs, labels)
                
            running_loss += loss.item()
            
            pred.append(outputs.cpu().detach())
            label.append(labels.cpu().detach())
            # break
        pred = torch.cat(pred)
        label = torch.cat(label)
        if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
            torch.save(pred, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
            torch.save(label, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')

    # softmax predictions
    pred = F.softmax(pred).to(device)
    # pred = pred[:, unq_classes]

    # # pad predictions and labels with `pad_n` true positives

    metric = MultilabelAveragePrecision(num_labels=CONFIG.num_classes, average="macro")
    valid_map = metric(pred.detach().cpu(), label.detach().cpu().long())
    
    #valid_map = metric(padded_preds, padded_labels)
    # calculate average precision
    # valid_map = average_precision_score(
    #     label.cpu().long(),
    #     pred.detach().cpu(),
    #     average='macro',
    # )
    # _, padded_preds = torch.max(padded_preds, 1)

    # acc = (padded_preds == padded_labels).sum().item() / len(padded_preds)
    # print("Validation Accuracy:", acc)
    
    
    print("Validation mAP:", valid_map)
    
    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/len(data_loader),
        "valid/map": valid_map,
        "custom_step": step,
        
    })
    
    return running_loss/len(data_loader), valid_map


def init_wandb(CONFIG: Dict[str, Any]):
    """ 
    Initialize the weights and biases logging
    """
    run = wandb.init(
        project="acoustic-species-reu2023",
        config=CONFIG,
        mode="disabled" if CONFIG.logging is False else "online"
    )
    run.name = (
        f"EFN-{CONFIG.epochs}"+
        f"-{CONFIG.train_batch_size}-{CONFIG.valid_batch_size}" +
        f"-{CONFIG.sample_rate}-{CONFIG.hop_length}-" +
        f"{CONFIG.max_time}-{CONFIG.n_mels}" +
        f"-{CONFIG.n_fft}-{CONFIG.seed}-" +
        run.name.split('-')[-1]
    )
    run.name = f"EFN-{CONFIG.epochs}-{CONFIG.train_batch_size}-{CONFIG.valid_batch_size}-{CONFIG.sample_rate}-{CONFIG.hop_length}-{CONFIG.max_time}-{CONFIG.n_mels}-{CONFIG.n_fft}-{CONFIG.seed}-" + run.name.split('-')[-1]
    return run

def load_datasets(CONFIG: Dict[str, Any]) \
    -> Tuple[PyhaDF_Dataset, PyhaDF_Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Loads datasets and dataloaders for train and validation
    """

    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs,
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader

def main():
    torch.multiprocessing.set_start_method('spawn')
    CONFIG = get_config()
    print(CONFIG)
    global wandb_run
    wandb_run = init_wandb(CONFIG)
    set_seed(CONFIG.seed)
    
    # Load in dataset
    print("Loading Dataset")
    train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets(CONFIG)
    
    print("Loading Model...")
    model_for_run = BirdCLEFModel(CONFIG=CONFIG).to(device)
    model_for_run.create_loss_fn(train_dataset)
    if CONFIG.model_checkpoint is not None:
        model_for_run.load_state_dict(torch.load(CONFIG.model_checkpoint))
    optimizer = Adam(model_for_run.parameters(), lr=CONFIG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Succesful :P")
    
    print("Training")
    step = 0
    best_valid_cmap = 0

    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        train_loss, step, best_valid_cmap = train(
            model_for_run, 
            train_dataloader,
            optimizer,
            scheduler,
            device,
            step,
            best_valid_cmap,
            epoch,
            CONFIG
        )
        
        valid_loss, valid_map = valid(model_for_run, val_dataloader, device, step, CONFIG)
        print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map}" )

        if valid_map > best_valid_cmap:
            path = os.path.join("models",wandb_run.name + '.pt')
            os.mkdir("models")
            torch.save(model_for_run.state_dict(), path)
            print("Model saved in:", path)
            print(f"Validation cmAP Improved - {best_valid_cmap} ---> {valid_map}")
            best_valid_cmap = valid_map

if __name__ == '__main__':
    main()
