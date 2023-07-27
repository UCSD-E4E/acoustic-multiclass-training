"""
Sweeps file: 
    Run it to start a new sweep or start another agent for an existing
sweep. If the former, set the sweep_id option in config
"""

import torch
import wandb
import torch

from pyha_analyzer import config
from pyha_analyzer.train import main as train_main

cfg = config.cfg


sweep_config = {
    'method'            : 'bayes',
    'name'              : 'sweep',
    'early_terminate'   : {'type': 'hyperband', 'min_iter': 3},
    'metric'            : {'goal': 'maximize', 'name' : 'valid/map'},
    'parameters'        : {
        # General
        'epochs' : {'values':[2]},
        'model' : {'values':["eca_nfnet_l0"]},
        'is_unchunked': {'values':[True]},
        'does_center_chunking': {'values':[False]},
        'overlap': {'max': 1.0, 'min': 0.},
        'min_length_s': {'max': 0.5, 'min': 0.},
        'n_fft': {'max':1024, 'min':256, 'distribution': 'int_uniform'},
        'chunk_length_s': {'max':7, 'min':1, 'distribution': 'int_uniform'},
        'n_mels': {'max':256, 'min':64, 'distribution': 'int_uniform'},
        'hop_length': {'max':512, 'min':64, 'distribution': 'int_uniform'},
    }
}

def main():
    """
    Main function
    """
    sweep_id = cfg.sweep_id
    wandb.login()
    sweep_project = f"{cfg.wandb_project}-sweep"
    if not sweep_id:
        print("Starting a new sweep")
        sweep_id = wandb.sweep(
            sweep_config,
            entity=cfg.wandb_entity,
            project=sweep_project)
    else:
        sweep_id = f"{cfg.wandb_entity}/{sweep_project}/{cfg.sweep_id}"
    
    for _ in range(cfg.agent_run_count):
        wandb.agent(sweep_id, function = train_main, count=1)
        print("HOPEFULLY AGENT IS DONE ================================")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main()
