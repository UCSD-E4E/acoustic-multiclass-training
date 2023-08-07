""" Gets the testing mAP of a model on soundscapes """

import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

from pyha_analyzer import config
from pyha_analyzer import dataset
from pyha_analyzer import train
from pyha_analyzer import utils
from pyha_analyzer.models.timm_model import TimmModel

cfg = config.cfg

def get_test_map():
    """ Returns the testing mAP for the specified data and class list """
    torch.multiprocessing.set_start_method('spawn')
    print(f"Device is: {cfg.device}, Preprocessing Device is {cfg.prepros_device}")
    utils.set_seed(cfg.seed)
    wandb.init(mode="disabled")
    
    # Get dataset
    df = pd.read_csv(cfg.dataframe_csv, index_col=0)
    if cfg.class_list is None:
        raise ValueError("Class list must be specified in config")
    test_ds = dataset.PyhaDFDataset(df,train=False, species=cfg.class_list)
    dataloader = DataLoader(
        test_ds,
        cfg.train_batch_size,
        shuffle=False,
        num_workers=cfg.jobs,
    )

    # Get model
    model_for_run = TimmModel(num_classes=test_ds.num_classes, 
                              model_name=cfg.model).to(cfg.device)
    model_for_run.create_loss_fn(test_ds)
    try:
        model_for_run.load_state_dict(torch.load(cfg.model_checkpoint))
    except FileNotFoundError as exc:
        raise FileNotFoundError("Model not found: " + cfg.model_checkpoint) from exc
    
    # Testing
    model_for_run.eval()
    valid_map, _ = train.valid(model_for_run, dataloader,None, 0.0, 1.0)
    return valid_map



if __name__ == "__main__":
    test_map = get_test_map()
    print(f"Test mAP: {test_map}")
    