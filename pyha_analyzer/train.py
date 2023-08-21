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
from typing import Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
import wandb

from pyha_analyzer import config
from pyha_analyzer.dataset import get_datasets, make_dataloaders, PyhaDFDataset
from pyha_analyzer.utils import set_seed
from pyha_analyzer.models.early_stopper import EarlyStopper
from pyha_analyzer.models.timm_model import TimmModel

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M')
cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

def run_batch(model: TimmModel,
                mels: torch.Tensor,
                labels: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Runs the model on a single batch 
        Args:
            model: the model to pass the batch through
            mels: single batch of input data
            labels: single batch of expecte output
        Returns (tuple of):
            loss: the loss of the batch
            outputs: the output of the model
    """
    mels = mels.to(cfg.device)
    labels = labels.to(cfg.device)
    if cfg.device == "cpu": 
        dtype = torch.bfloat16
    else: 
        dtype = torch.float16
    with autocast(device_type=cfg.device, dtype=dtype, enabled=cfg.mixed_precision):
        outputs = model(mels)
        loss = model.loss_fn(outputs, labels) # type: ignore
    outputs = outputs.to(dtype=torch.float32)
    loss = loss.to(dtype=torch.float32)
    return loss, outputs

def map_metric(outputs: torch.Tensor,
               labels: torch.Tensor,
               class_dist: torch.Tensor) -> Tuple[float, float]:
    """ Mean average precision metric for a batch of outputs and labels 
        Returns tuple of (class-wise mAP, sample-wise mAP) """
    metric = MultilabelAveragePrecision(num_labels=len(class_dist), average="none")
    out_for_score = outputs.detach().cpu()
    labels_for_score = labels.detach().cpu().long()
    map_by_class = metric(out_for_score, labels_for_score)
    cmap = map_by_class.nanmean()
    smap = (map_by_class * class_dist/class_dist.sum()).nansum()
    # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
    # Could be possible when model is untrained so we only have FNs
    if np.isnan(cmap):
        return 0, 0
    return cmap.item(), smap.item()

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def train(model: TimmModel,
        data_loader: DataLoader,
        valid_loader: DataLoader,
        infer_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        best_valid_cmap: float,
       ) -> float:
    """ Trains the model
    Returns new best valid map
    """
    logger.debug('size of data loader: %d', len(data_loader))
    model.train()

    log_n = 0
    log_loss = 0

    #scaler = torch.cuda.amp.GradScaler()
    start_time = datetime.datetime.now()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    log_pred = []
    log_labels = []

    for i, (mels, labels) in enumerate(data_loader):

        optimizer.zero_grad()

        loss, outputs = run_batch(model,  mels, labels)

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

        log_pred.append(F.sigmoid(outputs))
        log_labels.append(labels)
        log_loss += loss.item()
        log_n += 1

        if (i != 0 and i % (cfg.logging_freq) == 0) or i == len(data_loader) - 1:
            dataset: PyhaDFDataset = data_loader.dataset # type: ignore
            cmap, smap = map_metric(torch.cat(log_pred),torch.cat(log_labels),dataset.class_dist)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            start_time = datetime.datetime.now()
            annotations = ((i % cfg.logging_freq) or cfg.logging_freq) * cfg.train_batch_size
            #Log to Weights and Biases
            wandb.log({
                "train/loss": log_loss / log_n,
                "train/mAP": cmap,
                "train/smAP": smap,
                "i": i,
                "epoch": epoch,
                "clips/sec": annotations / duration,
                "epoch_progress": epoch + float(i)/len(data_loader),
            })
            logger.info("i: %s   epoch: %s   clips/s: %s   Loss: %s   cmAP: %s   smAP: %s",
                str(i).zfill(5),
                str(round(epoch+float(i)/len(data_loader),3)).ljust(5, '0'),
                str(round(annotations / duration,3)).ljust(7), 
                str(round(log_loss / log_n,3)).ljust(5), 
                str(round(cmap,3)).ljust(5),
                str(round(smap,3)).ljust(5)
            )
            log_loss = 0
            log_n = 0
            log_cmap = 0
            log_smap = 0

        if (i != 0 and i % (cfg.valid_freq) == 0):
            # Free memory so gpu is freed before validation run
            del mels
            del outputs
            del labels

            valid_start_time = datetime.datetime.now()
            _, best_valid_cmap = valid(model,
                                      valid_loader,
                                      infer_loader,
                                      epoch + i / len(data_loader),
                                      best_valid_cmap)
            model.train()
            # Ignore the time it takes to validate in annotations/sec
            start_time += datetime.datetime.now() - valid_start_time

    return best_valid_cmap

def valid(model: Any,
          data_loader: DataLoader,
          infer_loader: Optional[DataLoader],
          epoch_progress: float,
          best_valid_cmap: float = 1.0,
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

            loss, outputs = run_batch(model, mels, labels)
                
            running_loss += loss.item()
            
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())


    # softmax predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    dataset: PyhaDFDataset = data_loader.dataset # type: ignore
    cmap, smap = map_metric(log_pred, torch.cat(log_label), dataset.class_dist)

    # Log to Weights and Biases
    wandb.log({
        "valid/loss": running_loss/num_valid_samples,
        "valid/map": cmap,
        "valid/smAP": smap,
        "epoch_progress": epoch_progress,
    })

    logger.info("Validation Loss:\t%f\nValidation cmAP:\t%f\nValidation smAP:\t%f", 
                running_loss/len(data_loader),
                cmap, smap)

    if cmap > best_valid_cmap:
        logger.info("Model saved in: %s", save_model(model))
        logger.info("Validation cmAP Improved - %f ---> %f", best_valid_cmap, cmap)
        best_valid_cmap = cmap


    inference_valid(model, infer_loader, epoch_progress, cmap)
    return cmap, best_valid_cmap


def inference_valid(model: Any,
          data_loader: Optional[DataLoader],
          epoch_progress: float,
          valid_cmap: float):

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
            _, outputs = run_batch(model, mels, labels)
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())

    # sigmoid predictions
    log_pred = F.sigmoid(torch.cat(log_pred)).to(cfg.device)

    dataset: PyhaDFDataset = data_loader.dataset # type: ignore
    cmap, smap = map_metric(log_pred, torch.cat(log_label), dataset.class_dist)
    # Log to Weights and Biases
    domain_shift = np.abs(valid_cmap - cmap)
    wandb.log({
        "valid/domain_shift_diff": domain_shift,
        "valid/inferance_map": cmap,
        "valid/inference_smap": smap,
        "epoch_progress": epoch_progress,
    })

    logger.info("Infer cmAP: %f", cmap)
    logger.info("Infer smAP: %f", smap)
    logger.info("Domain Shift Difference: %f", domain_shift)

def save_model(model: TimmModel) -> str:
    """ Saves model in the models directory as a pt file, returns path """
    path = os.path.join("models", f"{cfg.model}-{time_now}.pt")
    if not os.path.exists("models"):
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
    train_dataset, val_dataset, infer_dataset = get_datasets()
    train_dataloader, val_dataloader, infer_dataloader = make_dataloaders(
        train_dataset, val_dataset, infer_dataset
    )

    logger.info("Loading Model...")
    model_for_run = TimmModel(num_classes=train_dataset.num_classes, 
                              model_name=cfg.model).to(cfg.device)
    model_for_run.create_loss_fn(train_dataset)
    if cfg.model_checkpoint != "":
        model_for_run.load_state_dict(torch.load(cfg.model_checkpoint))
    optimizer = Adam(model_for_run.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    
    logger.info("Training...")
    early_stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_valid_map_delta)

    best_valid_cmap = 0.0

    for epoch in range(cfg.epochs):
        logger.info("Epoch %d", epoch)

        best_valid_cmap = train(model_for_run,
                               train_dataloader,
                               val_dataloader,
                               infer_dataloader,
                               optimizer,
                               scheduler,
                               epoch,
                               best_valid_cmap)
        valid_cmap, best_valid_cmap = valid(model_for_run,
                                          val_dataloader,
                                          infer_dataloader,
                                          epoch + 1.0,
                                          best_valid_cmap)
        logger.info("Best validation cmAP: %f", best_valid_cmap)

        if cfg.early_stopping and early_stopper.early_stop(valid_cmap):
            logger.info("Early stopping has triggered on epoch %d", epoch)
            break

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main(in_sweep=False)
