# pylint: disable=R0902
# pylint: disable=W0621
# pylint: disable=R0913
# pylint: disable=E1121
# R0902, W0621 -> Not redefining these values, they are all the same value,
# R0913, E1121 -> These functions just have a lot of args that are frequently changed since ML

"""
    Contains the training and validation function and logging to Weights and Biases
    Methods:
        train: trains the model
        valid: calculates validation loss and accuracy
        set_seed: sets the random seed
        init_wandb: initializes the Weights and Biases logging


"""
from typing import Dict, Any, Tuple
import os
import datetime
from torchmetrics.classification import MultilabelAveragePrecision

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast
import numpy as np
from dataset import PyhaDF_Dataset, get_datasets
from model import TimmModel
from utils import set_seed, print_verbose
from config import get_config
from augmentations import LowpassFilter, RandomEQ, SyntheticNoise, BackgroundNoise
from tqdm import tqdm
import wandb



tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_run = None

def check_shape(outputs, labels):
    """
    Checks to make sure the output is the same
    """
    if outputs.shape != labels.shape:
        print(outputs.shape)
        print(labels.shape)
        raise RuntimeError("Shape diff between output of models and labels, see above and debug")



# Splitting this up would be annoying!!!
# pylint: disable=too-many-statements 
def train(model: Any,
        data_loader: PyhaDF_Dataset,
        valid_loader:  PyhaDF_Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        step: int,
        epoch: int,
        best_valid_cmap: float,
        CONFIG):
    """ Trains the model
        Returns:
            loss: the average loss over the epoch
            step: the current step
    """
    print_verbose('size of data loader:', len(data_loader),verbose=CONFIG.verbose)
    model.train()

    running_loss = 0
    log_n = 0
    log_loss = 0
    correct = 0
    total = 0
    mAP = 0
    
    scaler = torch.cuda.amp.GradScaler()


    start_time = datetime.datetime.now()
    for i, (mels, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        mels = mels.to(device)
        labels = labels.to(device)
        
        with autocast(device_type=device, dtype=torch.float16, enabled=CONFIG.mixed_precision):
            outputs = model(mels)
            check_shape(outputs, labels)
            loss = model.loss_fn(outputs, labels)
        outputs = outputs.to(dtype=torch.float32)
        loss = loss.to(dtype=torch.float32)

        if CONFIG.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        metric = MultilabelAveragePrecision(num_labels=model.num_classes, average="macro")
        batch_mAP = metric(outputs.detach().cpu(), labels.detach().cpu().long()).item()
        # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
        # Could be possible when model is untrained so we only have FNs
        if np.isnan(batch_mAP):
            batch_mAP = 0
        mAP += batch_mAP

        out_max_inx = torch.round(outputs)
        lab_max_inx = torch.round(labels)
        correct += (out_max_inx == lab_max_inx).sum().item()
        total += labels.shape[0] * labels.shape[1]

        log_loss += loss.item()
        log_n += 1

        if (i != 0 and i % (CONFIG.logging_freq) == 0) or i == len(data_loader) - 1:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            start_time = datetime.datetime.now()
            annotations = ((i % CONFIG.logging_freq) or CONFIG.logging_freq) * CONFIG.train_batch_size
            annotations_per_sec = annotations / duration
            epoch_progress = epoch + float(i) / len(data_loader)
            #Log to Weights and Biases
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/mAP": mAP / log_n,
                "train/accuracy": correct / total,
                "i": i,
                "epoch": epoch,
                "clips/sec": annotations_per_sec,
                "epoch_progress": epoch_progress,
            })
            print("i:", i, "epoch:", epoch_progress,
                  "clips/s:", annotations_per_sec, 
                  "Loss:", log_loss / log_n, 
                  "mAP", mAP / log_n)
            log_loss = 0
            log_n = 0
            correct = 0
            total = 0
            mAP = 0

        if (i != 0 and i % (CONFIG.valid_freq) == 0):
            _, _, best_valid_cmap = valid(model, valid_loader, step, best_valid_cmap, CONFIG)

        step += 1
    return running_loss/len(data_loader), best_valid_cmap


def valid(model: Any,
          data_loader: PyhaDF_Dataset,
          step: int,
          best_valid_cmap: float,
          CONFIG):
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
            check_shape(outputs, labels)

            loss = model.loss_fn(outputs, labels)

            running_loss += loss.item()

            pred.append(outputs.cpu().detach())
            label.append(labels.cpu().detach())

        pred = torch.cat(pred)
        label = torch.cat(label)
        if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
            torch.save(pred, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
            torch.save(label, "/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')

    # softmax predictions
    pred = F.softmax(pred).to(device)

    metric = MultilabelAveragePrecision(num_labels=model.num_classes, average="macro")
    valid_map = metric(pred.detach().cpu(), label.detach().cpu().long())

    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/len(data_loader),
        "valid/map": valid_map,
        "custom_step": step,
        "epoch_progress": step,
    })

    print(f"Validation Loss:\t{running_loss/len(data_loader)} \n Validation mAP:\t{valid_map}" )
    if valid_map > best_valid_cmap:
        path = os.path.join("models",wandb_run.name + '.pt')
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), path)
        print("Model saved in:", path)
        print(f"Validation cmAP Improved - {best_valid_cmap} ---> {valid_map}")
        best_valid_cmap = valid_map

    
    return running_loss/len(data_loader), valid_map, best_valid_cmap


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
        CONFIG.model + 
        f"-{time_now}"
    )

    return run

def load_datasets(train_dataset, val_dataset, CONFIG: Dict[str, Any]
        )-> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Loads datasets and dataloaders for train and validation
    """

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
    return train_dataloader, val_dataloader

def main():
    """ Main function
    """
    torch.multiprocessing.set_start_method('spawn')
    print("Device is: ",device)
    CONFIG = get_config()
    # Needed to redefine wandb_run as a global variable
    # pylint: disable=global-statement
    global wandb_run
    wandb_run = init_wandb(CONFIG)
    set_seed(CONFIG.seed)

    # Load in dataset
    print("Loading Dataset")
    # pylint: disable=unused-variable
    # for future can use torchvision.transforms.RandomApply here
    transforms = torch.nn.Sequential(SyntheticNoise("white", 0.05))
    train_dataset, val_dataset = get_datasets(transforms=transforms, CONFIG=CONFIG, alpha=0.3, mixup_idx=0)
    train_dataloader, val_dataloader = load_datasets(train_dataset, val_dataset, CONFIG)

    print("Loading Model...")
    model_for_run = TimmModel(num_classes=train_dataset.num_classes, 
                                model_name=CONFIG.model, 
                                CONFIG=CONFIG).to(device)
    model_for_run.create_loss_fn(train_dataset)
    if CONFIG.model_checkpoint is not None:
        model_for_run.load_state_dict(torch.load(CONFIG.model_checkpoint))
    optimizer = Adam(model_for_run.parameters(), lr=CONFIG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Successful :P")
    
    print("Training")
    step = 0
    best_valid_cmap = 0

    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        _, best_valid_cmap = train(
            model_for_run, 
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            device,
            step,
            epoch,
            best_valid_cmap,
            CONFIG
        )
        step += 1
        
        _, _, best_valid_cmap = valid(model_for_run, val_dataloader, step, best_valid_cmap, CONFIG)
        print("Best validation cmap:", best_valid_cmap.item())
        
if __name__ == '__main__':
    main()
