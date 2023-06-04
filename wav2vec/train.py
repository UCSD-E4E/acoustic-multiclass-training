# wav2vec
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification, get_scheduler, AdamW
from datasets import Dataset, load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
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
#from dataset import BirdCLEFDataset, get_datasets
#from model import BirdCLEFModel, GeM
from tqdm import tqdm
# # cmap metrics
# import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import MultilabelAveragePrecision
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=264, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=32, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=32, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)
parser.add_argument('-l', '--logging', default='True', type=str)
parser.add_argument('-lf', '--logging_freq', default=20, type=int)
parser.add_argument('-vf', '--valid_freq', default=1000, type=int)
parser.add_argument('-mch', '--model_checkpoint', default=None, type=str)
parser.add_argument('-md', '--map_debug', action='store_true')
parser.add_argument('-p', '--p', default=0, type=float, help='p for mixup')
parser.add_argument('-i', '--imb', action='store_true', help='imbalance sampler')
parser.add_argument('-pw', "--pos_weight", type=float, default=1, help='pos weight')
parser.add_argument('-lr', "--lr", type=float, default=3e-5, help='learning rate')
parser.add_argument('-bp', "--base_path", type=str, default=None)



#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training


def set_seed():
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)


def init_wandb(CONFIG):
    run = wandb.init(
        project="wav2vec",
        config=CONFIG,
        mode="disabled" if CONFIG.logging == False else "online"
    )
    run.name = f"EFN-{CONFIG.epochs}-{CONFIG.train_batch_size}-{CONFIG.valid_batch_size}-{CONFIG.sample_rate}-{CONFIG.hop_length}-{CONFIG.max_time}-{CONFIG.n_mels}-{CONFIG.n_fft}-{CONFIG.seed}-" + run.name.split('-')[-1]
    return run

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True
    )
    return inputs

if __name__ == '__main__':
    CONFIG = parser.parse_args()
    print(CONFIG)
    CONFIG.logging = True if CONFIG.logging == 'True' else False
    run = init_wandb(CONFIG)
    set_seed()
    
    base_path = '/share/acoustic_species_id/BirdCLEF2023_resampled_chunks'
    print("Loading Dataset")
    dataset = load_dataset("audiofolder", data_dir=base_path)
    # label processing
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # preprocessing
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    max_duration = 5.0
    
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)

    train_dataloader = torch.utils.data.DataLoader(
        encoded_dataset["train"],
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs,
    )
    val_dataloader = torch.utils.data.DataLoader(
        encoded_dataset["validation"],
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs
    )
    print("Loading Model...")
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", 
            num_labels=num_labels, 
            label2id=label2id, 
            id2label=id2label).to(device)

      
    optimizer = AdamW(model.parameters(), lr=3e-5)

    num_training_steps = CONFIG.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(device)
    print("Model / Optimizer Loading Successful :P")

    
    print("Training")
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(CONFIG.epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        torch.save(model.state_dict(), f"help-{epoch}.pt")
        metric= load_metric("accuracy")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        metric.compute()



    print(":o wow")