"""
Implementation of part of the data augmentation described in:
    https://arxiv.org/pdf/1604.07160.pdf
To use, simply import the RandomEQ class and use it as you would any other torch module
"""
from typing import Tuple
import torch
import torchaudio
#pylint: disable=E0402
from .. import config

def rand(low, high):
    """Wrapper for torch.rand""" 
    return (high - low) * torch.rand(1)[0] + low

class RandomEQ(torch.nn.Module):
    """
    Attributes: 
        f_range: tuple of upper and lower bounds for the frequency, in Hz
        g_range: tuple of upper and lower bounds for the gain, in dB
        q_range: tuple of upper and lower bounds for the Q factor
        num_applications: number of times to randomly EQ a part of the clip
        sample_rate: sampling rate of audio
    """
    def __init__(self,
                 f_range: Tuple[int, int] = (100, 6000),
                 g_range: Tuple[float, float] = (-8, 8),
                 q_range: Tuple[float, float] = (1, 9),
                 num_applications: int = 1):
        super().__init__()
        self.f_range = f_range
        self.g_range = g_range
        self.q_range = q_range
        self.num_applications = num_applications
        self.sample_rate = config.get_args("sample_rate")

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Randomly equalizes a part of the clip an arbitrary number of times
        Args: 
            clip: Tensor of audio data to be equalized

        Returns: Tensor of audio data with equalizations randomly applied 
        according to object parameters
        """
        for _ in range(self.num_applications):
            frequency = rand(*self.f_range)
            gain = rand(*self.g_range)
            Q = rand(*self.q_range)
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, Q)
        return clip
