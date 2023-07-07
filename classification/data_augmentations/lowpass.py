"""
Lowpass filter implemented as torch.nn.Module. To use, initialize,
and use as you would a regular torch.nn.Module
"""

import torch
import torchaudio
#pylint: disable=E0402
from .. import config

class LowpassFilter(torch.nn.Module):
    """
    Applies lowpass filter to audio based on provided parameters. 
    Note that due implementation details of the lowpass filters, 
    this may not work as expected for high q values (>5 ish)
    Attributes: 
        sample_rate: sample_rate of audio clip
        cutoff: cutoff frequency
        Q: Q value for lowpass filter
    """
    def __init__(self, cutoff: int, Q: float):
        super().__init__()
        self.sample_rate = config.get_args("sample_rate")
        self.cutoff = cutoff
        self.Q = Q

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Applies lowpass filter based on specified parameters
        Args:
            clip: Tensor of audio data

        Returns: Tensor of audio data with lowpass filter applied
        """
        return torchaudio.functional.lowpass_biquad(clip,
                                                    self.sample_rate,
                                                    self.cutoff,
                                                    self.Q)
