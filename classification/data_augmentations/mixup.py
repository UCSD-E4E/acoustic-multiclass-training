""" 
Contains class for applying mixup data augmentation. The class acts as a 
regular torch module, but cannot be used in a torch.nn.Sequential object.
Instead, use the mixup function as a wrapper, passing the other augmentations
to the mixup function in a torch.nn.Sequential object.
"""
from typing import Tuple, Callable
import torch
import numpy as np

class Mixup(torch.nn.Module):
    """
    Attributes: 
        dataset: Dataset from which to mixup with other clips
        alpha: Strength (proportion) of original audio in augmented clip
    """
    def __init__(self, dataset, alpha: float):
        super().__init__()
        self.dataset = dataset
        self.alpha = alpha

    def forward(
        self, clip: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip: Tensor of audio data
            target: Tensor representing label

        Returns: Tensor of audio data mixed with another randomly 
        chosen clip, Tensor of target mixed with the 
        target of the randomly chosen file
        """
        # Generate random index in dataset
        other_idx = np.random.randint(len(self.dataset))
        try:
            other_clip, other_target = self.dataset.get_clip(other_idx)
        except RuntimeError:
            print('Error loading other clip, mixup not performed')
            return clip, target
        mixed_clip = self.alpha * clip + (1 - self.alpha) * other_clip
        mixed_target = self.alpha * target + (1 - self.alpha) * other_target
        return mixed_clip, mixed_target


def mixup(sequential: torch.nn.Sequential, idx:int) -> Callable:
    """
    Wrapper function for mixup augmentation
    Args:
        sequential: Object containing all other data augmentations to 
        be performed
        idx: Index at which to perform mixup

    Returns: Function which applies all other data augmentations 
    as well as mixup, with order specified by idx
    """
    def helper(
        clip: torch.Tensor, target: torch.Tensor, dataset, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head = sequential[:idx]
        tail = sequential[idx:]
        clip = head(clip)
        clip, target = Mixup(dataset, alpha).forward(clip, target)
        clip = tail(clip)
        return clip, target

    return helper
