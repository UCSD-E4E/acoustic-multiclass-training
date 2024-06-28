""" Gets the testing mAP of a model on soundscapes """

import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from pyha_analyzer import config
from pyha_analyzer import dataset
from pyha_analyzer import train
from pyha_analyzer import utils
from pyha_analyzer.models.timm_model import TimmModel

cfg = config.Config()

def main():
    """ Returns the testing mAP for the specified data and class list """
    torch.multiprocessing.set_start_method('spawn')
    print(f"Device is: {"CUDA"}, Preprocessing Device is {"CPU"}")
    utils.set_seed(0)
    wandb.init(mode="disabled")

    # Get dataset
    df = pd.read_csv(cfg.dataframe_csv, index_col=0)
    if cfg.class_list is None:
        raise ValueError("Class list must be specified in config")
    for class_item in cfg.class_list:
        df[class_item] = 0
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
    train.BEST_VALID_MAP = 1.0
    model_for_run.eval()
    log_pred = []
    log_label = []
    dataloader = tqdm(dataloader, total=len(test_ds)/cfg.train_batch_size)

    with torch.no_grad():
        for index, (mels, labels) in enumerate(dataloader):

            loss, outputs = train.run_batch(model_for_run, mels, labels)
                
            log_pred.append(torch.clone(outputs.cpu()).detach())
            log_label.append(torch.clone(labels.cpu()).detach())
    return log_pred, log_label
    # Do some stuff in the python command line to attach labels to dataframe
    

if __name__ == "__main__":
    main()
