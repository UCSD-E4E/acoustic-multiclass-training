import pandas as pd
from pyha_analyzer import config
from pyha_analyzer.dataset import PyhaDFDataset
from pyha_analyzer.models.timm_model import TimmModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from mutagen import File
from pathlib import Path
from args import threshold, chunk_length, weights
import sqlite3
from datetime import datetime
cfg = config.cfg

def prob_to_int(p: float) -> int:
    return int(round(100 * float(p)))  # Annotation.speciesProbability saved as integer

def row_hits(row, classes):
    above = row[classes][row[classes] > threshold]
    if above.empty: return []
    off = row["OFFSET"]
    return [{"offset": int(off), "species": col, "confidence": float(val)}
            for col, val in above.items()]

def run_inference_on_filenames(base_dir: Path, 
                               file_rows, #file_rows: list of rows with keys 'recordingId' and 'filename'
                               classes, 
                               cfg):
    cfg.data_path = base_dir 
    rows = []
    audio_file_paths = []
    for r in file_rows:
        #filename = r["filename"]
        filename= Path(r["url"])
        full_path = base_dir / filename
        filename = str(filename)
        # print("full path is ", full_path)
        #full_path = r["url"]
        if not full_path.exists():
            print(f"[WARN] missing file: {full_path}")
            continue
        try:
            audio = File(full_path) 
        except Exception as e:
            print(f"Error reading {str(full_path)}: {e}")
            continue

        num_chunks = int(audio.info.length // chunk_length)
        for i in range(num_chunks):
            offset = i * chunk_length
            rows.append([offset, chunk_length, "amabaw1", filename])  # MANUAL ID kept as of now but eventaully clean up code to remove this as mandatory
        audio_file_paths.append(filename)
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=["OFFSET", "DURATION", "MANUAL ID", "FILE NAME"])

    infer_dataset = PyhaDFDataset(df, train=False, species=classes, cfg=cfg)
    loader = DataLoader(infer_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=0)
    
    #Load model
    # TODO Eventually: see if u can run load the model outside loop so that it only loads once. issue is that you need to have cfg.data_path and that changes per directory. u also need to have something in df so it (to me) seems like it needs to be there
    train_dataset = PyhaDFDataset(df, train=True, species=classes, cfg=cfg)
    model = TimmModel(num_classes=len(classes), model_name=cfg.model).to(cfg.device)
    model.create_loss_fn(train_dataset)
    checkpoint = torch.load(weights, map_location=cfg.device)
    model.load_state_dict(checkpoint)
    model.to(cfg.device)

    # Inference
    log_pred = []
    model.eval()
    with torch.no_grad():
        for mels, labels in loader:
            mels = mels.to(cfg.device)
            labels = labels.to(cfg.device)
            outputs = model(mels)
            outputs = outputs.to(dtype=torch.float32)
            log_pred.append(np.array(torch.clone(F.sigmoid(outputs).cpu()).detach()))
    log_pred = np.concatenate(log_pred)


    results = pd.DataFrame(log_pred, columns=infer_dataset.classes)
    results = pd.concat([infer_dataset.samples.reset_index(drop=True), results], axis=1)
    results["_hits"] = results.apply(lambda r: row_hits(r, infer_dataset.classes), axis=1)
    if "original_file_path" in results.columns:
        results["PT FILE NAME"] = results["FILE NAME"]
        results["FILE NAME"] = results["original_file_path"].fillna(results["FILE NAME"])
    return results