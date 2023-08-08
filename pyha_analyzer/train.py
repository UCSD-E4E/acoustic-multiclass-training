"""
    Contains the training and validation function and logging to Weights and Biases
    Methods:
        train: trains the model
        valid: calculates validation loss and accuracy
        set_seed: sets the random seed
"""
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

import wandb
from pyha_analyzer import config, dataset
from pyha_analyzer.models.early_stopper import EarlyStopper
from pyha_analyzer.models.timm_model import TimmModel
from pyha_analyzer.utils import set_seed, wandb_init

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M')
cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")


def map_metric(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """ Returns macro average precision metric for a batch of outputs and labels """
    metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
    out_for_score = outputs.detach().cpu()
    labels_for_score = labels.detach().cpu().long()
    map_out = metric(out_for_score, labels_for_score).item()
    # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
    # Could be possible when model is untrained so we only have FNs
    if np.isnan(map_out):
        return 0
    return map_out

class TrainingState():
    def __init__(self, model, train_dl, valid_dl, infer_dl):
        self.epoch = 1e-6
        self.optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-5, T_max=10)
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        self.best_valid_map = 0.
        self.valid_map = 0.
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.infer_dl = infer_dl
        self.early_stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_valid_map_delta)
        self.best_valid_map = 0.

    def log_metrics(self, i, start_time, log_n, log_loss, log_map):
        duration = (datetime.datetime.now() - start_time).total_seconds()
        start_time = datetime.datetime.now()
        annotations = ((i % cfg.logging_freq) or cfg.logging_freq) * cfg.train_batch_size
        #Log to Weights and Biases
        wandb.log({
            "train/loss": log_loss / log_n,
            "train/mAP": log_map / log_n,
            "i": i,
            "epoch": self.epoch,
            "clips/sec": annotations / duration,
            "epoch_progress": self.epoch + i/len(self.train_dl),
        })
        #TODO: Make pretty
        logger.info("i: %s   epoch: %s   clips/s: %s   Loss: %s   mAP: %s",
            str(i).zfill(5),
            str(round(self.epoch+float(i)/len(self.train_dl),3)).ljust(5, '0'),
            str(round(annotations / duration,3)).ljust(7), 
            str(round(log_loss / log_n,3)).ljust(5), 
            str(round(log_map / log_n,3)).ljust(5)
        )

    def run_epoch(self):
        logger.debug(f"size of data loader: {len(self.train_dl)}")
        self.model.train()
    
        log_n = log_loss = log_map = 0
    
        #scaler = torch.cuda.amp.GradScaler()
        start_time = datetime.datetime.now()
        start_epoch = self.epoch
        for i, (mels, labels) in enumerate(self.train_dl):
            self.epoch = start_epoch + i/len(self.train_dl)
            self.optimizer.zero_grad()
            loss, outputs = self.model.run_batch(mels, labels)
    
            if cfg.mixed_precision and cfg.device != "cpu":
                # Pyright complains about scaler.scale(loss) returning iterable of unknown types
                # Problem in the pytorch typing, documentation says it returns iterables of Tensors
                #  keep if needed - noqa: reportGeneralTypeIssues
                scaler.scale(loss).backward()  # type: ignore
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if cfg.mixed_precision:
                    logger.warning("cuda required, mixed precision not applied")
                loss.backward()
                self.optimizer.step()
                #TODO: Check if you can actually keep this
                #if self.scheduler is not None:
                self.scheduler.step()
    
            log_pred = F.sigmoid(outputs)
            log_map += map_metric(log_pred, labels, self.model.num_classes)
            log_loss += loss.item()
            log_n += 1
    
            #Log and reset metrics
            if (i != 0 and i % (cfg.logging_freq) == 0) or i == len(self.train_dl) - 1:
                self.log_metrics(i, start_time, log_n, log_loss, log_map)
                log_n = log_loss = log_map = 0
    
            # Free memory so gpu is freed before validation run
            if (i != 0 and i % (cfg.valid_freq) == 0):
                del mels
                del outputs
                del labels
    
                valid_start_time = datetime.datetime.now()
                self.valid()
                self.model.train()
                # Ignore the time it takes to validate in annotations/sec
                start_time += datetime.datetime.now() - valid_start_time
    
        return self.best_valid_map


    def valid(self):
        self.model.eval()
    
        running_loss = 0
        log_pred, log_label = [], []
        dataset_ratio: float = cfg.valid_dataset_ratio
        if self.epoch.is_integer():
            dataset_ratio = 1.0
    
        num_valid_samples = int(len(self.valid_dl)*dataset_ratio)
    
        # tqdm is a progress bar
        dl_iter = tqdm(self.valid_dl, position=5, total=num_valid_samples)
    
        with torch.no_grad():
            for index, (mels, labels) in enumerate(dl_iter):
                if index > num_valid_samples:
                    # Stop early if not doing full validation
                    break
    
                loss, outputs = self.model.run_batch(mels, labels)
                    
                running_loss += loss.item()
                
                log_pred.append(torch.clone(outputs.cpu()).detach())
                log_label.append(torch.clone(labels.cpu()).detach())
    
    
        # softmax predictions
        log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)
    
        self.valid_map = map_metric(log_pred, torch.cat(log_label), self.model.num_classes)
    
        # Log to Weights and Biases
        wandb.log({
            "valid/loss": running_loss/num_valid_samples,
            "valid/map": self.valid_map,
            "epoch_progress": self.epoch,
        })
    
        logger.info(f"Validation Loss: {running_loss/len(self.valid_dl)}\n"
                    f"Validation mAP:  {self.valid_map}")
    
        if self.valid_map > self.best_valid_map:
            logger.info("Model saved in: %s", utils.save_model(self.model))
            logger.info("Validation mAP Improved - %f ---> %f", self.best_valid_map, self.valid_map)
            self.best_valid_map = self.valid_map
    
    
        self.inference_valid()
        return self.valid_map, self.best_valid_map

    #TODO: Do validation first for pseudolabeling
    def inference_valid(self):
    
        """ Test Domain Shift To Soundscapes
    
        """
        if self.infer_dl is None:
            return
    
        self.model.eval()
    
        log_pred, log_label = [], []
    
        num_valid_samples = int(len(self.infer_dl))
    
        # tqdm is a progress bar
        dl_iter = tqdm(self.infer_dl, position=5, total=num_valid_samples)
    
        with torch.no_grad():
            for _, (mels, labels) in enumerate(dl_iter):
                _, outputs = self.model.run_batch(mels, labels)
                log_pred.append(torch.clone(outputs.cpu()).detach())
                log_label.append(torch.clone(labels.cpu()).detach())
    
        # sigmoid predictions
        log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)
    
        infer_map = map_metric(log_pred, torch.cat(log_label), self.model.num_classes)
        # Log to Weights and Biases
        domain_shift = np.abs(self.valid_map - infer_map)
        wandb.log({
            "valid/domain_shift_diff": domain_shift,
            "epoch_progress": self.epoch,
        })
    
        logger.info(f"Domain Shift Difference:\t{domain_shift}")


def main(in_sweep=True) -> None:
    """ Main function
    """
    logger.info(f"Device is: {cfg.device}\n"
                f"Preprocessing Device is {cfg.prepros_device}")
    set_seed(cfg.seed)
    wandb_init(in_sweep)

    # Load in dataset
    logger.info("Loading Dataset...")
    train_ds, valid_ds, infer_ds = dataset.get_datasets()
    train_dl, valid_dl, infer_dl = (
            dataset.get_dataloader(train_ds, valid_ds, infer_ds)
    )

    logger.info("Loading Model...")
    model = TimmModel(
            num_classes=train_ds.num_classes, model_name=cfg.model
    ).to(cfg.device)
    model.create_loss_fn(train_ds)

    if model.try_load_checkpoint():
        logger.info("Loaded model from checkpoint...")

    logger.info("Initializing training process...")
    train_process = TrainingState(model, valid_dl, train_dl, infer_dl)

    logger.info("Training...")
    for _ in range(cfg.epochs):
        train_process.run_epoch()

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main(in_sweep=False)
