"""
Instead, use the mixup function as a wrapper, passing the other augmentations
to the mixup function in a torch.nn.Sequential object.
"""
import os
import logging
from pathlib import Path
from typing import Tuple, Callable, Dict, Any
import numpy as np
import pandas as pd
import torch
import torchaudio

import utils
import config

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

class Mixup(torch.nn.Module):
    """
    Attributes:
        dataset: Dataset from which to mixup with other clips
        alpha_range: Range of alpha parameter, which determines 
        proportion of new audio in augmented clip
        p: Probability of mixing
    """
    # pylint: disable-next=too-many-arguments
    def __init__(
            self, 
            df: pd.DataFrame, 
            class_to_idx: Dict[str, Any],
            sample_rate: int,
            target_num_samples: int,
            alpha_range: Tuple[float,float]=(0.1, 0.4), 
            p: float=0.4,
            ):
        super().__init__()
        self.df = df
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.target_num_samples = target_num_samples
        self.alpha_range = alpha_range
        self.prob = p

    def forward(self,
        clip: torch.Tensor,
        target: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip: Tensor of audio data
            target: Tensor representing label

        Returns: Tensor of audio data mixed with another randomly
        chosen clip, Tensor of target mixed with the
        target of the randomly chosen file
        """
        alpha = utils.rand(*self.alpha_range)
        if utils.rand(0,1) < self.prob:
            return clip, target

        # Generate random index in dataset
        other_idx = utils.randint(0, len(self.df))
        try:
            other_clip, other_target = utils.get_annotation(
                    df = self.df,
                    index = other_idx, 
                    class_to_idx = self.class_to_idx, 
                    sample_rate = self.sample_rate, 
                    target_num_samples = self.target_num_samples, 
                    device = clip.device)
        except RuntimeError:
            logger.error('Error loading other clip, mixup not performed')
            return clip, target
        mixed_clip = (1 - alpha) * clip + alpha * other_clip
        mixed_target = (1 - alpha) * target + alpha * other_target
        return mixed_clip, mixed_target


def add_mixup(
        clip: torch.Tensor, 
        target: torch.Tensor, 
        mixup: Mixup, 
        sequential: torch.nn.Sequential, 
        idx:int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        clip: Tensor of audio data
        target: Tensor representing label
        sequential: Object containing all other data augmentations to
        be performed
        idx: Index at which to perform mixup

    Returns: clip, target with all transforms and mixup applied
    """
    head = sequential[:idx]
    tail = sequential[idx:]
    clip = head(clip)
    clip, target = mixup(clip, target)
    clip = tail(clip)
    return clip, target

def gen_noise(num_samples: int, psd_shape_func: Callable) -> torch.Tensor:
    """
    Args:
        num_samples: length of noise Tensor to generate
        psd_shape_func: function that gives the shape of the noise's
        power spectrum distribution

    Returns: noise Tensor of length num_samples
    """
    #Reverse fourier transfrom of random array to get white noise
    white_signal = torch.fft.rfft(torch.rand(num_samples))
    # Adjust frequency amplitudes according to
    # function determining the psd shape
    shape_signal = psd_shape_func(torch.fft.rfftfreq(num_samples))
    # Normalize signal
    shape_signal = shape_signal / torch.sqrt(torch.mean(shape_signal.float()**2))
    # Adjust frequency amplitudes according to noise type
    noise = white_signal * shape_signal
    return torch.fft.irfft(noise).to('cuda')

def noise_generator(func: Callable):
    """
    Given PSD shape function, returns a new function that takes in parameter N
    and generates noise Tensor of length N
    """
    return lambda N: gen_noise(N, func)

@noise_generator
def white_noise(vec: torch.Tensor):
    """White noise PSD shape"""
    return torch.ones(vec.shape) 

@noise_generator
def blue_noise(vec: torch.Tensor):
    """Blue noise PSD shape"""
    return torch.sqrt(vec)

@noise_generator
def violet_noise(vec: torch.Tensor):
    """Violet noise PSD shape"""
    return vec

@noise_generator
def brown_noise(vec: torch.Tensor):
    """Brown noise PSD shape"""
    return 1/torch.where(vec == 0, float('inf'), vec)

@noise_generator
def pink_noise(vec: torch.Tensor):
    """Pink noise PSD shape"""
    return 1/torch.where(vec == 0, float('inf'), torch.sqrt(vec))

# For some reason this class can't be printed in the repl,
# but works fine in scripts?
class SyntheticNoise(torch.nn.Module):
    """
    Attributes:
        noise_type: type of noise to add to clips
        alpha: Strength (proportion) of noise audio in augmented clip
    """
    noise_names = {'pink': pink_noise,
                   'brown': brown_noise,
                   'violet': violet_noise,
                   'blue': blue_noise,
                   'white': white_noise}
    def __init__(self, noise_type: str, alpha: float):
        super().__init__()
        self.noise_type = noise_type
        self.alpha = alpha
    def forward(self, clip: torch.Tensor)->torch.Tensor:
        """
        Args:
            clip: Tensor of audio data

        Returns: Clip mixed with noise according to noise_type and alpha
        """
        noise_function = self.noise_names[self.noise_type]
        noise = noise_function(len(clip))
        return (1 - self.alpha) * clip + self.alpha* noise


class RandomEQ(torch.nn.Module):
    """
    Implementation of part of the data augmentation described in:
        https://arxiv.org/pdf/1604.07160.pdf
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
        self.sample_rate = cfg.sample_rate

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Randomly equalizes a part of the clip an arbitrary number of times
        Args:
            clip: Tensor of audio data to be equalized

        Returns: Tensor of audio data with equalizations randomly applied
        according to object parameters
        """
        for _ in range(self.num_applications):
            frequency = utils.rand(*self.f_range)
            gain = utils.rand(*self.g_range)
            q_val = utils.rand(*self.q_range)
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, q_val)
        return clip

class BackgroundNoise(torch.nn.Module):
    """
    torch module for adding background noise to audio tensors
    Attributes:
        alpha: Strength (proportion) of original audio in augmented clip
        sample_rate: Sample rate (Hz)
        length: Length of audio clip (s)
    """
    def __init__(self,
            alpha: float,
            length=5,
            norm=True
            ):
        super().__init__()
        self.noise_path = Path(cfg.background_path)
        self.alpha = alpha
        self.sample_rate = cfg.sample_rate
        self.length = length
        self.norm = norm
        if self.noise_path != "":
            files = list(os.listdir(self.noise_path))
            audio_extensions = (".mp3",".wav",".ogg",".flac",".opus",".sphere")
            self.noise_clips = [f for f in files if f.endswith(audio_extensions)]
            

    def forward(self, clip: torch.Tensor)->torch.Tensor:
        """
        Mixes clip with noise chosen from noise_path
        Args:
            clip: Tensor of audio data

        Returns: Tensor of original clip mixed with noise
        """
        # Skip loading if no noise path
        if self.noise_path == "":
            return clip
        # If loading fails, skip for now
        try:
            noise_clip = self.choose_random_noise()
        except RuntimeError:
            logger.warning('Error loading noise clip, background noise augmentation not performed')
            return clip
        return self.alpha*clip + (1-self.alpha)*noise_clip

    def choose_random_noise(self):
        """
        Returns: Tensor of random noise, loaded from self.noise_path
        """
        rand_idx = utils.randint(0, len(self.noise_clips))
        noise_file = self.noise_path / self.noise_clips[rand_idx]
        clip_len = self.sample_rate * self.length

        # pryright complains that load isn't called from torchaudio. It is.
        waveform, sample_rate = torchaudio.load(noise_file) #pyright: ignore
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        if self.norm:
            waveform = utils.norm(waveform)
        start_idx = utils.randint(0, len(waveform) - clip_len)
        return waveform[start_idx, start_idx+clip_len]


class LowpassFilter(torch.nn.Module):
    """
    Applies lowpass filter to audio based on provided parameters.
    Note that due implementation details of the lowpass filters,
    this may not work as expected for high q values (>5 ish)
    Attributes:
        sample_rate: sample_rate of audio clip
        cutoff: cutoff frequency
        q_val: Q value for lowpass filter
    """
    def __init__(self, cutoff: int, q_val: float):
        super().__init__()
        self.sample_rate = cfg.sample_rate
        self.cutoff = cutoff
        self.q_val = q_val

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
                                                    self.q_val)
