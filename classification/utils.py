""" 
    Stores useful functions for the classification module 
    Methods:
        print_verbose: prints the arguments if verbose is set to True

"""

from operator import itemgetter

import numpy as np
import torch
from config import get_config


def print_verbose(*args, **kwargs):
    """ 
        Prints the arguments if verbose is set to True
    """
    if("verbose" in kwargs and kwargs["verbose"]):
        del kwargs["verbose"]
        print(*args, **kwargs)

def set_seed(seed: int):
    """ Sets numpy and pytorch seeds to the CONFIG.seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def norm(tensor: torch.Tensor):
    """ 
    Normalizes the tensor
    """
    return tensor/torch.linalg.vector_norm(tensor, ord=2)

def randint(low: int, high: int) -> int:
    """
    Returns a random integer between [low, high)
    """
    return int(torch.randint(low, high, (1,)))

def rand(low: float , high: float) -> float:
    """
    Returns a random float between [low, high)
    """
    return (high - low) * float(torch.rand(1)[0]) + low

def get_args(*args):
    """
    Args:
        *args: Series of strings corresponding to the command line arguments
    Returns: Values of the command line arguments
    """
    CONFIG = vars(get_config())
    return itemgetter(*args)(CONFIG)
    
