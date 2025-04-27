#script for running several training runs with different configurations
import time
import torch
import json
import pyha_analyzer as pa
from pyha_analyzer import config

cfg = config.cfg

# Function to update the config and run the training
def run_training(config_dict):
    # Dynamically update cfg variables from the config_dict
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"Warning: Config has no attribute '{key}', skipping...")

    # Set multiprocessing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')

    # Run the training
    pa.train.main(in_sweep=False)

# Function to load configurations from a JSON file and run the training runs
def run_training_runs(config_file):
    # Load configurations from the JSON file
    with open(config_file, 'r') as file:
        configurations = json.load(file)

    # Iterate over each configuration and run the training
    for config in configurations:
        print(f"Running training for: {config.get('wandb_run_name', 'Unnamed Run')}")
        run_training(config)
        time.sleep(10)  # Optional wait time between runs to make sure logging is done fully

# Entry point to start the training runs
if __name__ == "__main__":
    config_file_path = "sequential_run_cfg"  # Change to your config file path if needed
    run_training_runs(config_file_path)

