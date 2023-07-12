""" 
    Stores useful functions for the classification module 
    Methods:
        print_verbose: prints the arguments if verbose is set to True

"""

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

def norm(s):
    """ 
    Normalizes the tensor
    """
    return s/torch.linalg.vector_norm(s, ord=2)

def randint(low, high):
    """
    Returns a random integer between [low, high)
    """
    return int(torch.randint(low, high, (1,)))

def rand(low, high):
    """
    Returns a random float between [low, high)
    """
    return (high - low) * float(torch.rand(1)[0]) + low
