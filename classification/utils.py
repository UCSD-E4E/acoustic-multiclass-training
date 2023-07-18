""" 
    Stores useful functions for the classification module 
    Methods:

"""

import numpy as np
import torch

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
