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
from pyha_analyzer import config, dataset, pseudolabel
from pyha_analyzer.dataset import PyhaDFDataset
from pyha_analyzer.models.early_stopper import EarlyStopper
from pyha_analyzer.models.timm_model import TimmModel
from pyha_analyzer.utils import set_seed

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

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def train(model: TimmModel,
        data_loader: DataLoader,
        valid_loader: DataLoader,
        infer_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        best_valid_map: float,
       ) -> float:
    """ 
    Trains the model
    Returns new best valid map
    """
    logger.debug('size of data loader: %d', len(data_loader))
    model.train()

    log_n = 0
    log_loss = 0
    log_map = 0

    #scaler = torch.cuda.amp.GradScaler()
    start_time = datetime.datetime.now()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for i, (mels, labels) in enumerate(data_loader):

        optimizer.zero_grad()

        loss, outputs = model.run_batch(mels, labels)

        if cfg.mixed_precision and cfg.device != "cpu":
            # Pyright complains about scaler.scale(loss) returning iterable of unknown types
            # Problem in the pytorch typing, documentation says it returns iterables of Tensors
            #  keep if needed - noqa: reportGeneralTypeIssues
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.mixed_precision:
                logger.warning("cuda required, mixed precision not applied")
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        log_pred = F.sigmoid(outputs)
        log_map += map_metric(log_pred, labels, model.num_classes)
        log_loss += loss.item()
        log_n += 1

        if (i != 0 and i % (cfg.logging_freq) == 0) or i == len(data_loader) - 1:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            start_time = datetime.datetime.now()
            annotations = ((i % cfg.logging_freq) or cfg.logging_freq) * cfg.train_batch_size
            #Log to Weights and Biases
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/mAP": log_map / log_n,
                "i": i,
                "epoch": epoch,
                "clips/sec": annotations / duration,
                "epoch_progress": epoch + float(i)/len(data_loader),
            })
            logger.info("i: %s   epoch: %s   clips/s: %s   Loss: %s   mAP: %s",
                str(i).zfill(5),
                str(round(epoch+float(i)/len(data_loader),3)).ljust(5, '0'),
                str(round(annotations / duration,3)).ljust(7), 
                str(round(log_loss / log_n,3)).ljust(5), 
                str(round(log_map / log_n,3)).ljust(5)
            )
            log_loss = 0
            log_n = 0
            log_map = 0

        if (i != 0 and i % (cfg.valid_freq) == 0):
            # Free memory so gpu is freed before validation run
            del mels
            del outputs
            del labels

            valid_start_time = datetime.datetime.now()
            _, best_valid_map = valid(model,
                                      valid_loader,
                                      infer_loader,
                                      epoch + i / len(data_loader),
                                      best_valid_map)
            model.train()
            # Ignore the time it takes to validate in annotations/sec
            start_time += datetime.datetime.now() - valid_start_time

    return best_valid_map

def valid(model: Any,
          data_loader: DataLoader,
          infer_loader: Optional[DataLoader],
          epoch_progress: float,
          best_valid_map: float = 1.0,
          ) -> Tuple[float, float]:
    """ Run a validation loop
    Arguments:
        model: the model to validate
        data_loader: the validation data loader
        epoch_progress: the progress of the epoch
            - Note: If this is an integer, it will run the full
                    validation set, otherwise runs cfg.valid_dataset_ratio
    Returns:
        Loss
    """
    model.eval()

    running_loss = 0
    log_pred, log_label = [], []
    dataset_ratio: float = cfg.valid_dataset_ratio
    if float(epoch_progress).is_integer():
        dataset_ratio = 1.0

    num_valid_samples = int(len(data_loader)*dataset_ratio)

    # tqdm is a progress bar
    dl_iter = tqdm(data_loader, position=5, total=num_valid_samples)

    with torch.no_grad():
        for index, (mels, labels) in enumerate(dl_iter):
            if index > num_valid_samples:
                # Stop early if not doing full validation
                break

            loss, outputs = model.run_batch(mels, labels)
                
            running_loss += loss.item()
            
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())


    # softmax predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    valid_map = map_metric(log_pred, torch.cat(log_label), model.num_classes)

    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/num_valid_samples,
        "valid/map": valid_map,
        "epoch_progress": epoch_progress,
    })

    logger.info("Validation Loss:\t%f\nValidation mAP:\t%f", 
                running_loss/len(data_loader),
                valid_map)

    if valid_map > best_valid_map:
        logger.info("Model saved in: %s", save_model(model))
        logger.info("Validation mAP Improved - %f ---> %f", best_valid_map, valid_map)
        best_valid_map = valid_map


    inference_valid(model, infer_loader, epoch_progress, valid_map)
    return valid_map, best_valid_map


def inference_valid(model: Any,
          data_loader: Optional[DataLoader],
          epoch_progress: float,
          valid_map: float):

    """ Test Domain Shift To Soundscapes

    """
    if data_loader is None:
        return

    model.eval()

    log_pred, log_label = [], []

    num_valid_samples = int(len(data_loader))

    # tqdm is a progress bar
    dl_iter = tqdm(data_loader, position=5, total=num_valid_samples)

    with torch.no_grad():
        for _, (mels, labels) in enumerate(dl_iter):
            _, outputs = model.run_batch(mels, labels)
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())

    # sigmoid predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    infer_map = map_metric(log_pred, torch.cat(log_label), model.num_classes)
    # Log to Weights and Biases
    domain_shift = np.abs(valid_map - infer_map)
    wandb.log({
        "valid/domain_shift_diff": domain_shift,
        "epoch_progress": epoch_progress,
    })

    logger.info("Domain Shift Difference:\t%f", domain_shift)
    

def save_model(model: TimmModel) -> str:
    """ Saves model in the models directory as a pt file, returns path """
    path = Path("models")/(f"{cfg.model}-{time_now}.pt")
    if not Path("models").exists():
        os.mkdir("models")
    torch.save(model.state_dict(), path)
    return path

def set_name(run):
    """
    Set wandb run name
    """
    if cfg.wandb_run_name == "auto":
        # This variable is always defined
        cfg.wandb_run_name = cfg.model # type: ignore
    run.name = f"{cfg.wandb_run_name}-{time_now}"
    return run

def logging_setup() -> None:
    """ Setup logging on the main process
    Display config information
    """
    file_handler = logging.FileHandler("recent.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.debug("Debug logging enabled")
    logger.debug("Config: %s", cfg.config_dict)
    logger.debug("Git hash: %s", cfg.git_hash)

def run_train(model, 
              train_dataloader, 
              val_dataloader, 
              infer_dataloader, 
              optimizer,
              scheduler, 
              epochs,
    ):
    """ 
    Convenience wrapper for model training 
    """
    early_stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_valid_map_delta)

    best_valid_map = 0.0

    for epoch in range(epochs):
        logger.info("Epoch %d", epoch)

        best_valid_map = train(model,
                               train_dataloader,
                               val_dataloader,
                               infer_dataloader,
                               optimizer,
                               scheduler,
                               epoch,
                               best_valid_map)
        valid_map, best_valid_map = valid(model,
                                          val_dataloader,
                                          infer_dataloader,
                                          epoch + 1.0,
                                          best_valid_map)
        logger.info("Best validation map: %f", best_valid_map)
        if cfg.early_stopping and early_stopper.early_stop(valid_map):
            logger.info("Early stopping has triggered on epoch %d", epoch)
            break

def main(in_sweep=True) -> None:
    """ Main function
    """
    logger.info("Device is: %s, Preprocessing Device is %s", cfg.device, cfg.prepros_device)
    set_seed(cfg.seed)

    if in_sweep:
        run = wandb.init()
        for key, val in dict(wandb.config).items():
            setattr(cfg, key, val)
        wandb.config.update(cfg.config_dict)
    else:
        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            config=cfg.config_dict,
            mode="online" if cfg.logging else "disabled")
        set_name(run)

    # Load in dataset
    logger.info("Loading Dataset...")
    train_dataset, val_dataset, infer_dataset = dataset.get_datasets()
    train_dataloader, val_dataloader, infer_dataloader = (
            dataset.get_dataloader(train_dataset, val_dataset, infer_dataset)
    )

    logger.info("Loading Model...")
    model = TimmModel(
            num_classes=train_dataset.num_classes, model_name=cfg.model
    ).to(cfg.device)
    model.create_loss_fn(train_dataset)

    if model.try_load_checkpoint():
        logger.info("Loaded model from checkpoint")
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    
    logger.info("Training...")
    run_train(model,
              train_dataloader,
              val_dataloader,
              infer_dataloader,
              optimizer,
              scheduler,
              cfg.epochs)

    if cfg.pseudo:
        pseudo_labels(model, optimizer, scheduler, run.name)

def pseudo_labels(model, optimizer, scheduler, run_name):
    """
    Fine tune on pseudo labels
    """
    run = wandb.init(
            entity=cfg.wandb_entity,
            project=f"{cfg.wandb_project}-pseudo",
            config=cfg.config_dict,
            mode="online" if cfg.logging else "disabled")
    run.name = run_name

    logger.info("Loading pseudo labels...")
    raw_df = pseudolabel.make_raw_df()
    predictions = pseudolabel.run_raw(model, raw_df)
    pseudo_df = pseudolabel.get_pseudolabels(
            predictions, raw_df, cfg.pseudo_threshold
    )
    train_dataset = PyhaDFDataset(pseudo_df, train=cfg.pseudo_data_augs, species=cfg.class_list)
    # Note that this is just the same data as the train dataset
    val_dataset = PyhaDFDataset(pseudo_df, train=False, species=cfg.class_list)
    _, _, infer_ds = dataset.get_datasets()
    train_dataloader, val_dataloader, infer_dataloader = (
        dataset.get_dataloader(train_dataset, val_dataset, infer_dataset)
    )
    logger.info("Finetuning on pseudo labels...")
    run_train(model,
          train_dataloader,
          val_dataloader,
          infer_dataloader,
          optimizer,
          scheduler,
          cfg.epochs)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main(in_sweep=False)
