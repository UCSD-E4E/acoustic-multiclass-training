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
from tqdm import tqdm
from train import load_datasets, valid

tqdm.pandas()
time_now  = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device is: ",device)
CONFIG = get_config()

train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets(CONFIG)
    
print("Loading Model...")
model_for_run = TimmModel(num_classes=130, 
                            model_name="convnextv2_nano", 
                            checkpoint="/share/acoustic_species_id/models/convnextv2_nano-20230710-1731-0.pt",
                            CONFIG=CONFIG).to(device)

model_for_run.load_state_dict(torch.load("/share/acoustic_species_id/models/convnextv2_nano-20230710-1731-0.pt"))

valid(model_for_run, val_dataloader, 0, 1, CONFIG)