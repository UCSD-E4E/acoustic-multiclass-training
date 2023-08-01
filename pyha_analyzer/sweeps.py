"""
Sweeps file: 
    Run it to start a new sweep or start another agent for an existing
sweep. If the former, set the sweep_id option in config
"""
import logging

import torch
import yaml

import wandb
from pyha_analyzer import config
from pyha_analyzer.train import main as train_main

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

def main():
    """
    Main function
    """
    sweep_id = cfg.sweep_id
    wandb.login()
    sweep_project = f"{cfg.wandb_project}-sweep"
    if not sweep_id:
        logger.info("Starting a new sweep")
        try:
            with open("sweeps.yml", 'r', encoding="utf-8") as sweep_file:
                sweep_config = yaml.safe_load(sweep_file)
        except FileNotFoundError:
            logger.info("sweeps.yml not found, loading default sweep config")
            with open("documentation/default_sweeps.yml", 'r', encoding="utf-8") as sweep_file:
                sweep_config = yaml.safe_load(sweep_file)

        sweep_id = wandb.sweep(
            sweep_config,
            entity=cfg.wandb_entity,
            project=sweep_project)
    else:
        sweep_id = f"{cfg.wandb_entity}/{sweep_project}/{cfg.sweep_id}"
    
    for _ in range(cfg.agent_run_count):
        wandb.agent(sweep_id, function = train_main, count=1)
        logger.info("HOPEFULLY AGENT IS DONE ================================")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    main()
