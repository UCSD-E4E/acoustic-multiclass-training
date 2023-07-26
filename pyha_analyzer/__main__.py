""" Entry point for training script """
import torch
from pyha_analyzer import train

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    train.main()
