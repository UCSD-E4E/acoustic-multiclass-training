""" Contains methods related to pseudo-labelling
The basic idea behind pseudo-labelling is to take un labelled data, run it through a model,
and use its confident predictions as training data. 
This will hopefully help with domain shift problems because we can train on soundscape data.
"""
import os
from pathlib import Path

import pandas as pd
import torchaudio
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb

from pyha_analyzer import config
from pyha_analyzer import dataset
from pyha_analyzer import utils
from pyha_analyzer.models.timm_model import TimmModel

cfg = config.cfg

def make_raw_df() -> pd.DataFrame:
    """ Returns dataframe of all raw chunks in {data_path}/pseudo """
    files = os.listdir(os.path.join(cfg.data_path, "pseudo"))
    valid_formats = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
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
                "OFFSET": i * cfg.chunk_length_s,
                "DURATION": cfg.chunk_length_s,
                cfg.manual_id_col: cfg.config_dict["class_list"][0], # Set to stop error
                "FILE NAME": os.path.join("pseudo", file),
                "CLIP LENGTH": file_len}))
    return pd.DataFrame(chunks)

def run_raw(model: TimmModel, df: pd.DataFrame):
    """ Returns predictions tensor from raw chunk dataframe """
    # Get dataset
    if cfg.config_dict["class_list"] is None:
        raise ValueError("Pseudolabelling requires class list")
    raw_ds = dataset.PyhaDFDataset(df,train=False, species=cfg.config_dict["class_list"])
    dataloader = DataLoader(raw_ds, cfg.train_batch_size, num_workers=cfg.jobs)

    # Testing
    model.eval()
    log_pred = []
    dataloader = tqdm(dataloader, total=len(raw_ds)/cfg.train_batch_size)

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
    raw_df.assign(**{cfg.manual_id_col: filtered_species})
    raw_df.assign(CONFIDENCE=confidence)
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

def main():
    """ Main function """
    torch.multiprocessing.set_start_method('spawn', force=True)
    print(f"Device is: {cfg.device}, Preprocessing Device is {cfg.prepros_device}")
    wandb.init(mode="disabled")
    utils.set_seed(cfg.seed)

    print("Generating raw dataframe...")
    raw_df = make_raw_df()
    print("Running model...")
    model = TimmModel(num_classes=len(cfg.class_list), model_name=cfg.model).to(cfg.device)
    model.create_loss_fn(None)
    if not model.try_load_checkpoint():
        raise RuntimeError("No model checkpoint found")
    pred = run_raw(model, raw_df)
    print("Generating pseudolabels...")
    pseudo = get_pseudolabels(pred, raw_df, threshold=0.7)
    cur_data = pd.read_csv(cfg.dataframe_csv, index_col=0)
    merged = merge_with_cur(cur_data, pseudo)
    print(f"Current dataset has {cur_data.shape[0]} rows")
    print(f"Pseudo label dataset has {pseudo.shape[0]} rows")
    merged.to_csv(cfg.dataframe_csv + "_pseudo.csv")
    print("Saved at " + cfg.dataframe_csv + "_pseudo.csv")
    
if __name__ == "__main__":
    main()
    