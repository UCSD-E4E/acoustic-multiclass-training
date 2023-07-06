import torch
import torchaudio
from typing import Tuple

def rand(low, high):
    return (high - low) * torch.rand(1)[0] + low

# Implementation of part of the data augmentation described in:
# https://arxiv.org/pdf/1604.07160.pdf
# Randomly equalizes a part of the clip an arbitrary number of times
class RandomEQ(torch.nn.Module):
    def __init__(self,
                 f_range: Tuple[int, int] = (100, 6000),
                 g_range: Tuple[float, float] = (-8, 8),
                 q_range: Tuple[float, float] = (1, 9),
                 num_applications: int = 1,
                 sample_rate: int = 44100):
        super().__init__()
        self.f_range = f_range
        self.g_range = g_range
        self.q_range = q_range
        self.num_applications = num_applications
        self.sample_rate = sample_rate

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_applications):
            frequency = rand(*self.f_range)
            gain = rand(*self.g_range)
            Q = rand(*self.q_range)
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, Q)
        return clip
