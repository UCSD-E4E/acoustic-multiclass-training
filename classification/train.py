"""
    Contains the training and validation function and logging to Weights and Biases
    Methods:
        train: trains the model
        valid: calculates validation loss and accuracy
        set_seed: sets the random seed
        init_wandb: initializes the Weights and Biases logging


"""
import datetime
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from augmentations import SyntheticNoise
from config import get_config
from dataset import PyhaDFDataset, get_datasets
from model import EarlyStopper, TimmModel
from torch.amp import autocast
from torch.optim import Adam
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
from utils import print_verbose, set_seed

import wandb

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def train(model: Any,
        data_loader: PyhaDFDataset,
        valid_loader:  PyhaDFDataset,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        best_valid_map: float,
        CONFIG) -> Tuple[float, int, float]:
    """ Trains the model
        Returns:
            loss: the average loss over the epoch
            best_valid_map: the best validation mAP
    """
    print_verbose('size of data loader:', len(data_loader),verbose=CONFIG.verbose)
    model.train()

    running_loss = 0
    log_n = 0
    log_loss = 0
    mAP = 0
    
    scaler = torch.cuda.amp.GradScaler()

    start_time = datetime.datetime.now()
    
    scaler = torch.cuda.amp.GradScaler()

    for i, (mels, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        mels = mels.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=CONFIG.mixed_precision):
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
        batch_map = metric(outputs.detach().cpu(), labels.detach().cpu().long()).item()
        # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
        # Could be possible when model is untrained so we only have FNs
        if np.isnan(batch_map):
            batch_map = 0
        m_ap += batch_map

        log_loss += loss.item()
        log_n += 1

        if (i != 0 and i % (CONFIG.logging_freq) == 0) or i == len(data_loader) - 1:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            start_time = datetime.datetime.now()
            batches = (i % CONFIG.logging_freq) or CONFIG.logging_freq
            annotations = batches * CONFIG.train_batch_size
            annotations_per_sec = annotations / duration
            epoch_progress = epoch + float(i) / len(data_loader)
            #Log to Weights and Biases
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/mAP": mAP / log_n,
                "i": i,
                "epoch": epoch,
                "clips/sec": annotations_per_sec,
                "epoch_progress": epoch_progress,
            })
            print("i:", i, "epoch:", epoch_progress,
                  "clips/s:", annotations_per_sec, 
                  "Loss:", log_loss / log_n, 
                  "mAP", m_ap / log_n)
            log_loss = 0
            log_n = 0
            mAP = 0

        if (i != 0 and i % (CONFIG.valid_freq) == 0):
            valid_start_time = datetime.datetime.now()
            _, _, best_valid_map = valid(model, 
                                          valid_loader, 
                                          epoch + i / len(data_loader), 
                                          best_valid_map, 
                                          CONFIG)
            # Ignore the time it takes to validate in annotations/sec
            start_time += datetime.datetime.now() - valid_start_time
    return running_loss/len(data_loader), best_valid_map


# pylint: disable=too-many-locals
def valid(model: Any,
          data_loader: PyhaDF_Dataset,
          epoch_progress: float,
          best_valid_map: float,
          CONFIG) -> Tuple[float, float, float]:
    """ Run a validation loop
    Arguments:
        model: the model to validate
        data_loader: the validation data loader
        epoch_progress: the progress of the epoch
            - Note: If this is an integer, it will run the full
                    validation set, otherwise runs config.valid_dataset_ratio
        best_valid_map: the best validation mAP
        CONFIG
    Returns:
        Tuple of (loss, valid_map, best_valid_map)
    """
    model.eval()

    running_loss = 0
    pred = []
    label = []
    dataset_ratio: float = CONFIG.valid_dataset_ratio
    if epoch_progress.is_integer():
        dataset_ratio = 1.0

    # tqdm is a progress bar
    dl = tqdm(data_loader, position=5, total=int(len(data_loader)*dataset_ratio))

    if CONFIG.map_debug and CONFIG.model_checkpoint is not None:
        pred = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/pred.pt')
        label = torch.load("/".join(CONFIG.model_checkpoint.split('/')[:-1]) + '/label.pt')
    else:
        with torch.no_grad():
            for index, (mels, labels) in enumerate(dl):
                if index > len(dl) * dataset_ratio:
                    # Stop early if not doing full validation
                    break
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
    pred = F.softmax(pred).to(DEVICE)

    metric = MultilabelAveragePrecision(num_labels=model.num_classes, average="macro")
    valid_map = metric(pred.detach().cpu(), label.detach().cpu().long())

    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/len(data_loader),
        "valid/map": valid_map,
        "epoch_progress": epoch_progress,
    })

    print(f"Validation Loss:\t{running_loss/len(data_loader)} \n Validation mAP:\t{valid_map}" )
    if valid_map > best_valid_map:
        path = os.path.join("models",CONFIG.model +f"-{time_now}" + '.pt')
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), path)
        print("Model saved in:", path)
        print(f"Validation mAP Improved - {best_valid_map} ---> {valid_map}")
        best_valid_map = valid_map

    
    return running_loss/len(data_loader), valid_map, best_valid_map


def init_wandb(CONFIG: Dict[str, Any]):
    """
    Initialize the weights and biases logging
    """
    run = wandb.init(
        project="acoustic-species-reu2023",
        entity="acoustic-species",
        config=CONFIG,
        mode="online" if CONFIG.logging else "disabled"
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
    print("Device is: ",DEVICE)
    CONFIG = get_config()
    init_wandb(CONFIG)
    set_seed(CONFIG.seed)

    # Load in dataset
    print("Loading Dataset")
    # for future can use torchvision.transforms.RandomApply here
    transforms = torch.nn.Sequential(SyntheticNoise("white", 0.05))
    train_dataset, val_dataset = get_datasets(transforms=transforms, CONFIG=CONFIG)
    train_dataloader, val_dataloader = load_datasets(train_dataset, val_dataset, CONFIG)

    print("Loading Model...")
    model_for_run = TimmModel(num_classes=train_dataset.num_classes, 
                                model_name=CONFIG.model, 
                                CONFIG=CONFIG).to(DEVICE)
    model_for_run.create_loss_fn(train_dataset)
    if CONFIG.model_checkpoint is not None:
        model_for_run.load_state_dict(torch.load(CONFIG.model_checkpoint))
    optimizer = Adam(model_for_run.parameters(), lr=CONFIG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Successful :P")
    
    print("Training")
    best_valid_map = 0
    early_stopper = EarlyStopper(patience=CONFIG.patience, min_delta=CONFIG.min_delta)
    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        _, best_valid_map = train(
            model_for_run,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            epoch,
            best_valid_map,
            CONFIG
        )
        _, valid_map, best_valid_map = valid(model_for_run, 
                                             val_dataloader, 
                                             epoch + 1.0, 
                                             best_valid_map, 
                                             CONFIG)

        print("Best validation map:", best_valid_map.item())
        if CONFIG.early_stopping and early_stopper.early_stop(valid_map):
            print("Early stopping has triggered on epoch", epoch)
            break

        
if __name__ == '__main__':
    main()
