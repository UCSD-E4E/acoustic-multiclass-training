"""
Sweeps file: 
    Run it to start a new sweep or start another agent for an existing
sweep. If the former, set the sweep_id option in config
"""

import torch
import wandb

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
        'epochs' : {'values':[3]},
        'rand_eq_p'         : {'max': 0.6, 'min': 0.},
        'rand_eq_min_f'     : {'max': 5000, 'min': 20},
        'rand_eq_f_addition': {'max': 10000, 'min': 4000},
        'rand_eq_min_g'     : {'max': 0, 'min': -8},
        'rand_eq_g_addition': {'max': 16, 'min': 4},
        'rand_eq_min_q'     : {'max': 4., 'min': 0.5},
        'rand_eq_q_addition': {'max': 8., 'min': 2.},
        'rand_eq_iters'     : {'max': 5, 'min': 1},
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
