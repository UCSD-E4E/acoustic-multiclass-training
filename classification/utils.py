""" 
    Stores useful functions for the classification module 
    Methods:
        print_verbose: prints the arguments if verbose is set to True

"""

from typing import Dict, Any
import numpy as np
import torch

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
