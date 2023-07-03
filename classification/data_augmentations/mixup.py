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
        other_idx = np.random.randint(0, len(self.dataset))
        other_clip, other_target = self.dataset.get_clip(other_idx)
        mixed_clip = self.alpha * clip + (1 - self.alpha) * other_clip
        mixed_target = self.alpha * target + (1 - self.alpha) * other_target
        return mixed_clip, mixed_target


def mixup_wrapper(sequential: torch.nn.Sequential, idx) -> Callable:
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
