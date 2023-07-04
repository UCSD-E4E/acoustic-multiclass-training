""" 
Contains class for applying mixup data augmentation. The class acts as a 
regular torch module, but cannot be used in a torch.nn.Sequential object.
Instead, use the mixup function as a wrapper, passing the other augmentations
to the mixup function in a torch.nn.Sequential object.
"""
import sys
from typing import Tuple, Callable
import numpy as np
import torch
sys.path.append('..')
from dataset import PyhaDF_Dataset

class Mixup(torch.nn.Module):
    def __init__(self, dataset: PyhaDF_Dataset, alpha: float):
        super().__init__()
        self.dataset = dataset
        self.alpha = alpha

    # TODO: Check chosen clip is not same
    def forward(
        self, clip: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        other_idx = np.random.randint(len(self.dataset))
        try:
            other_clip, other_target = self.dataset.get_clip(other_idx)
        except:
            print('Error loading other clip, mixup not performed')
            return clip, target
        mixed_clip = self.alpha * clip + (1 - self.alpha) * other_clip
        mixed_target = self.alpha * target + (1 - self.alpha) * other_target
        return mixed_clip, mixed_target


def mixup(sequential: torch.nn.Sequential, idx) -> Callable:
    def helper(
        clip: torch.Tensor, target: torch.Tensor, dataset: PyhaDF_Dataset, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head = sequential[:idx]
        tail = sequential[idx:]
        clip = head(clip)
        clip, target = Mixup(dataset, alpha).forward(clip, target)
        clip = tail(clip)
        return clip, target

    return helper
