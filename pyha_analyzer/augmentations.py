"""
File containing data augmentations implemented as torch.nn.Module
Each augmentation is initialized with only a Config object
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import torch
import torchaudio

from pyha_analyzer import config, utils

logger = logging.getLogger("acoustic_multiclass_training")

def invert(seq: Iterable[int]) -> List[float]:
    """
    Replace each element in list with its inverse
    """
    if 0 in seq: 
        raise ValueError('Passed iterable cannot contain zero')
    return [1/x for x in seq]

def hyperbolic(seq: Iterable[int]) -> List[Tuple[float, int]]:
    """
    Takes a list of numbers and assigns them a probability
    distribution accourding to the inverse of their values
    """
    invert_seq = invert(seq)
    norm_factor = sum(invert_seq)
    probabilities = [x/norm_factor for x in invert_seq]
    return list(zip(probabilities, seq))

def sample(distribution: List[Tuple[float, int]]) -> int:
    """
    Sample single value from distribution given by list of tuples
    """
    probabilities, values = zip(*distribution)
    return np.random.choice(values, p = probabilities)

def gen_uniform_values(n: int, min_value=0.05) -> List[float] :
    """
    Generates n values uniformly such that their sum is 1
    Args:
        n: number of values to generate, must be at least two
        min_value: Minimum possible value in list. Must be less than 1/(n-1)
    Returns: List of n values
    """
    step = 1/(n-1)
    rand_points = np.arange(0, 1, step = step)
    rand_points = [0.] + [p + utils.rand(0, step-min_value) for p in rand_points]
    alphas = (
        [1 - rand_points[-1]] +
        [rand_points[i] - rand_points[i-1] for i in range(1, n)]
    )
    assert sum(alphas) <=1.00005
    assert sum(alphas) >=0.99995
    return alphas

class Mixup(torch.nn.Module):
    """
    Attributes:
        dataset: Dataset from which to mixup with other clips
        alpha_range: Range of alpha parameter, which determines 
        proportion of new audio in augmented clip
        p: Probability of mixing
    """
    def __init__(
            self, 
            df: pd.DataFrame, 
            class_to_idx: Dict[str, Any],
            cfg: config.Config
            ):
        super().__init__()
        self.df = df
        self.class_to_idx = class_to_idx
        self.prob = cfg.mixup_p
        self.cfg = cfg
        self.ceil_interval = cfg.mixup_ceil_interval
        self.min_alpha = cfg.mixup_min_alpha

        # Get probability distribution for how many clips to mix
        possible_num_clips = list(range(
                cfg.mixup_num_clips_range[0],
                cfg.mixup_num_clips_range[1] + 1))
        self.num_clips_distribution = hyperbolic(possible_num_clips)

    def get_rand_clip(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get random clip from self.df
        """
        idx = utils.randint(0, len(self.df))
        try:
            clip, target = utils.get_annotation(
                    df = self.df,
                    index = idx, 
                    conf = self.cfg,
                    class_to_idx = self.class_to_idx)
            return clip, target
        except RuntimeError:
            logger.error('Error loading other clip, ommitted from mixup')
            return None

    def mix_clips(self, 
                  clip: torch.Tensor, 
                  target: torch.Tensor, 
                  other_annotations: List[Tuple[torch.Tensor, torch.Tensor]]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup clips and targets of clip, target, other_annotations
        """
        annotations = other_annotations + [(clip, target)]
        clips, targets = zip(*annotations)
        mix_factors = gen_uniform_values(len(annotations), min_value = self.min_alpha)

        mixed_clip = sum(c * f for c, f in zip(clips, mix_factors))
        mixed_target = sum(t * f for t, f in zip(targets, mix_factors))
        assert isinstance(mixed_target, torch.Tensor)
        assert isinstance(mixed_clip, torch.Tensor)
        assert mixed_clip.shape == clip.shape
        assert mixed_target.shape == target.shape
        mixed_target = utils.ceil(mixed_target, interval = self.ceil_interval)
        return mixed_clip, mixed_target

    def forward(
            self,
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
        if utils.rand(0,1) >= self.prob:
            return clip, target

        num_other_clips = sample(self.num_clips_distribution)
        other_annotations = [self.get_rand_clip() for _ in range(num_other_clips)]
        other_annotations = list(filter(None, other_annotations))
        return self.mix_clips(clip, target, other_annotations)


def gen_noise(num_samples: int, psd_shape_func: Callable) -> torch.Tensor:
    """
    Args:
        num_samples: length of noise Tensor to generate
        psd_shape_func: function that gives the shape of the noise's
        power spectrum distribution
        device: CUDA or CPU for processing

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
    return torch.fft.irfft(noise)

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
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.noise_type = cfg.noise_type
        self.alpha = cfg.noise_alpha
        self.device = cfg.prepros_device

    def forward(self, clip: torch.Tensor)->torch.Tensor:
        """
        Args:
            clip: Tensor of audio data

        Returns: Clip mixed with noise according to noise_type and alpha
        """
        noise_function = self.noise_names[self.noise_type]
        noise = noise_function(len(clip)).to(self.device)
        return (1 - self.alpha) * clip + self.alpha* noise


class RandomEQ(torch.nn.Module):
    """
    Implementation of part of the data augmentation described in:
        https://arxiv.org/pdf/1604.07160.pdf
    Attributes:
        f_range: tuple of upper and lower bounds for the frequency, in Hz
        g_range: tuple of upper and lower bounds for the gain, in dB
        q_range: tuple of upper and lower bounds for the Q factor
        iterations: number of times to randomly EQ a part of the clip
        sample_rate: sampling rate of audio
    """
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.f_range = cfg.rand_eq_f_range
        self.g_range = cfg.rand_eq_g_range
        self.q_range = cfg.rand_eq_q_range
        self.iterations = cfg.rand_eq_iters
        self.sample_rate = cfg.sample_rate

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Randomly equalizes a part of the clip an arbitrary number of times
        Args:
            clip: Tensor of audio data to be equalized

        Returns: Tensor of audio data with equalizations randomly applied
        according to object parameters
        """
        for _ in range(self.iterations):
            frequency = utils.log_rand(*self.f_range)
            gain = utils.rand(*self.g_range)
            q_val = utils.rand(*self.q_range)
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, q_val)
        return clip

# Mald about it pylint!
# pylint: disable-next=too-many-instance-attributes
class BackgroundNoise(torch.nn.Module):
    """
    torch module for adding background noise to audio tensors
    Attributes:
        alpha: Strength (proportion) of original audio in augmented clip
        sample_rate: Sample rate (Hz)
        length: Length of audio clip (s)
    """
    def __init__(self, cfg: config.Config, norm=False):
        super().__init__()
        self.noise_path = Path(cfg.bg_noise_path)
        self.noise_path_str = cfg.bg_noise_path
        self.alpha_range = cfg.bg_noise_alpha_range
        self.sample_rate = cfg.sample_rate
        self.length = cfg.chunk_length_s
        self.device = cfg.prepros_device
        self.norm = norm
        if self.noise_path_str != "" and cfg.bg_noise_p > 0.0:
            files = list(os.listdir(self.noise_path))
            audio_extensions = (".mp3",".wav",".ogg",".flac",".opus",".sphere",".pt")
            self.noise_clips = [f for f in files if f.endswith(audio_extensions)]
            if len(self.noise_clips) == 0:
                raise RuntimeError("Background noise path specified, but no audio files found. " \
                                   + "Check supported format list in augmentations.py")
        elif cfg.bg_noise_p!=0.0:
            raise RuntimeError("Background noise probability is non-zero, "
            + "yet no background path was specified. Please update config.yml")
        else:
            pass # Background noise is disabled if p=0 and path=""
            

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Mixes clip with noise chosen from noise_path
        Args:
            clip: Tensor of audio data

        Returns: Tensor of original clip mixed with noise
        """
        # Skip loading if no noise path
        alpha = utils.rand(*self.alpha_range)
        if self.noise_path_str == "":
            return clip
        # If loading fails, skip for now
        try:
            noise_clip = self.choose_random_noise()
        except RuntimeError as e:
            logger.warning('Error loading noise clip, background noise augmentation not performed')
            logger.error(e)
            return clip
        return (1 - alpha)*clip + alpha*noise_clip

    def choose_random_noise(self):
        """
        Returns: Tensor of random noise, loaded from self.noise_path
        """
        rand_idx = utils.randint(0, len(self.noise_clips))
        noise_file = self.noise_path / self.noise_clips[rand_idx]
        clip_len = self.sample_rate * self.length

        if str(noise_file).endswith(".pt"):
            waveform = torch.load(noise_file).to(self.device, dtype=torch.float32)/32767.0
        else:
            # pryright complains that load isn't called from torchaudio. It is.
            waveform, sample_rate = torchaudio.load(noise_file, normalize=True) #pyright: ignore
            waveform = waveform[0].to(self.device)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(
                        waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
                torch.save((waveform*32767).to(dtype=torch.int16), noise_file.with_suffix(".pt"))
                os.remove(noise_file)
                file_name = self.noise_clips[rand_idx]
                self.noise_clips.remove(file_name)
                self.noise_clips.append(str(Path(file_name).with_suffix(".pt").name))
        if self.norm:
            waveform = utils.norm(waveform)
        start_idx = utils.randint(0, len(waveform) - clip_len)
        return waveform[start_idx:start_idx+clip_len]


class LowpassFilter(torch.nn.Module):
    """
    Applies lowpass filter to audio based on provided parameters.
    Note that due implementation details of the biquad filters,
    this may not work as expected for high q values (>5 ish)
    Attributes:
        sample_rate: sample_rate of audio clip
        cutoff: cutoff frequency
        q_val: Q value for lowpass filter
    """
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.sample_rate = cfg.sample_rate
        self.cutoff = cfg.lowpass_cutoff
        self.q_val = cfg.lowpass_q_val

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

class HighpassFilter(torch.nn.Module):
    """
    Applies highpass filter to audio based on provided parameters.
    Note that due implementation details of the biquad filters,
    this may not work as expected for high q values (>5 ish)
    Attributes:
        sample_rate: sample_rate of audio clip
        cutoff: cutoff frequency
        q_val: Q value for highpass filter
    """
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.sample_rate = cfg.sample_rate
        self.cutoff = cfg.highpass_cutoff
        self.q_val = cfg.highpass_q_val

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Applies lowpass filter based on specified parameters
        Args:
            clip: Tensor of audio data

        Returns: Tensor of audio data with lowpass filter applied
        """
        return torchaudio.functional.highpass_biquad(clip,
                                                    self.sample_rate,
                                                    self.cutoff,
                                                    self.q_val)
