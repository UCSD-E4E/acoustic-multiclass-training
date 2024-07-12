
# %%
import pandas as pd
from pyha_analyzer.train import run_batch
from pyha_analyzer.dataset import get_datasets, make_dataloaders, PyhaDFDataset
from pyha_analyzer import config
from pyha_analyzer.models.timm_model import TimmModel
import logging

logger = logging.getLogger("acoustic_multiclass_training")

cfg = config.cfg

import torch
model_for_run = TimmModel(num_classes=132, 
                            model_name=cfg.model).to(cfg.device)

print(next(model_for_run.parameters()).device)

for i in range(torch.cuda.device_count()):
    device_name = f'cuda:{i}'
    print(f'{i} device name:{torch.cuda.get_device_name(torch.device(device_name))}')