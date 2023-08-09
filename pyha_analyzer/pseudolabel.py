""" Contains methods related to pseudo-labeling
The basic idea behind pseudo-labeling is to take un labelled data, run it through a model,
and use its confident predictions as training data. 
This will hopefully help with domain shift problems because we can train on soundscape data.
"""
import logging
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyha_analyzer import config, dataset, utils
from pyha_analyzer.models.timm_model import TimmModel
from pyha_analyzer.train import TrainProcess

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

def make_raw_df() -> pd.DataFrame:
    """ Returns dataframe of all raw chunks in {data_path}/pseudo """
    files = os.listdir(os.path.join(cfg.data_path, "pseudo"))
    valid_formats = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
    valid_formats += tuple(f.upper() for f in valid_formats)
    # Split into raw chunks
    chunks = []
    for file in tqdm(files):
        file_len = 0
        path = Path(file)
        if path.suffix not in valid_formats:
            continue
        # Check if there is a preprocessed version
        if str(path.with_suffix(".pt")) in files:
            audio = torch.load(Path(cfg.data_path) / "pseudo" / path.with_suffix(".pt"))
            file_len = audio.shape[0] / cfg.sample_rate
        # Else load formatted version
        else:
            # Pyright is stupid and can't find torchaudio.load
            audio, sample_rate = \
                torchaudio.load(Path(cfg.data_path) / "pseudo" / file) # type: ignore
            file_len = audio.shape[1] / sample_rate
        # Append chunks to dataframe
        for i in range(int(file_len/cfg.chunk_length_s)):
            chunks.append(pd.Series({
                cfg.offset_col: i * cfg.chunk_length_s,
                cfg.duration_col: cfg.chunk_length_s,
                cfg.manual_id_col: cfg.config_dict["class_list"][0], # Set to stop error
                cfg.file_name_col: os.path.join("pseudo", file),
                "CLIP LENGTH": file_len}))
    return pd.DataFrame(chunks)

def run_raw(model: TimmModel, df: pd.DataFrame):
    """ Returns predictions tensor from raw chunk dataframe """
    # Get dataset
    if cfg.config_dict["class_list"] is None:
        raise ValueError("Pseudo-labeling requires class list")
    raw_ds = dataset.PyhaDFDataset(df,train=False, species=cfg.config_dict["class_list"])
    dataloader = DataLoader(raw_ds, cfg.train_batch_size, num_workers=cfg.jobs)

    # Testing
    model.eval()
    log_pred = []
    dataloader = tqdm(dataloader, total=math.ceil(len(raw_ds)/cfg.train_batch_size))

    with torch.no_grad():
        for _, (mels, labels) in enumerate(dataloader):
            _, outputs = model.run_batch(mels, labels)
            log_pred.append(torch.clone(outputs.cpu()).detach())
    return torch.nn.functional.sigmoid(torch.concat(log_pred))


def get_pseudolabels(pred: torch.Tensor, raw_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """ Returns dataframe of pseudolabels from predictions and raw dataframe """
    df = pd.DataFrame(pred, columns=cfg.class_list, dtype='float64')
    allowed = df.apply(lambda x: x.max() > threshold, axis=1)
    filtered_species = df.idxmax(axis=1)[allowed]
    confidence = df.max(axis=1)[allowed]
    raw_df = raw_df[allowed]
    raw_df[cfg.manual_id_col] = filtered_species
    raw_df["CONFIDENCE"] = confidence
    return raw_df

def merge_with_cur(annotations: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    """ Merge psuedolabel dataset with current dataset and save to new file """
    # Merge with current dataset
    out = pd.concat([annotations, pseudo_df])
    out.reset_index(drop=True,inplace=True)
    # Delete columns with missing values to prevent row deletion errors
    out.dropna(inplace=True, axis='columns')
    return out

def add_pseudolabels(model: TimmModel, cur_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """ Return annotations dataframe merged with pseudolabels """
    raw_df = make_raw_df()
    pred = run_raw(model, raw_df)
    pseudo_df = get_pseudolabels(pred, raw_df, threshold)
    print(f"Current dataset has {cur_df.shape[0]} rows")
    print(f"Pseudo label dataset has {pseudo_df.shape[0]} rows")
    return merge_with_cur(cur_df, pseudo_df)

    
def pseudo_labels(model):
    """
    Fine tune on pseudo labels
    """
    logger.info("Generating raw dataframe...")
    raw_df = make_raw_df()
    logger.info("Running model...")
    predictions = run_raw(model, raw_df)
    logger.info("Generating pseudo labels...")
    pseudo_df = get_pseudolabels(
        predictions, raw_df, cfg.pseudo_threshold
    )
    model.create_loss_fn(pseudo_df)
    pseudo_df.to_csv("tmp_pseudo_labels.csv")
    logger.info("Saved pseudo dataset to tmp_pseudo_labels.csv")
    print(f"Pseudo label dataset has {pseudo_df.shape[0]} rows")

    logger.info("Loading dataset...")
    train_ds = dataset.PyhaDFDataset(
        pseudo_df, train=cfg.pseudo_data_augs, species=cfg.class_list
    )
    _, valid_ds, infer_ds = dataset.get_datasets()
    train_dl, valid_dl, infer_dl = (
        dataset.get_dataloader(train_ds, valid_ds, infer_ds)
    )

    logger.info("Finetuning on pseudo labels...")
    train_process = TrainProcess(model, train_dl, valid_dl, infer_dl)
    train_process.valid()
    for _ in range(cfg.epochs):
        train_process.run_epoch()



def main(in_sweep=True):
    """ Main function """
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"Device is: {cfg.device}, Preprocessing Device is {cfg.prepros_device}")
    utils.logging_setup()
    utils.set_seed(cfg.seed)
    utils.wandb_init(in_sweep)
    print("Creating model...")
    model = TimmModel(num_classes=len(cfg.class_list), model_name=cfg.model).to(cfg.device)
    model.create_loss_fn(None)
    if not model.try_load_checkpoint():
        raise RuntimeError("No model checkpoint found")
    pseudo_labels(model)


if __name__ == "__main__":
    main(in_sweep=False)
