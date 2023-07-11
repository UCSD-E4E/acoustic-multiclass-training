"""
Instead, use the mixup function as a wrapper, passing the other augmentations
to the mixup function in a torch.nn.Sequential object.
"""
import os
from pathlib import Path
from typing import Tuple, Callable
import torch
import utils
from config import get_config


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
        other_idx = utils.randint(0, len(self.dataset))
        try:
            other_clip, other_target = self.dataset.get_clip(other_idx, apply_transforms=False)
        except RuntimeError:
            print('Error loading other clip, mixup not performed')
            return clip, target
        mixed_clip = self.alpha * clip + (1 - self.alpha) * other_clip
        mixed_target = self.alpha * target + (1 - self.alpha) * other_target
        return mixed_clip, mixed_target


def add_mixup(mixup: Mixup, sequential: torch.nn.Sequential, idx:int) -> Callable:
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
        clip: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head = sequential[:idx]
        tail = sequential[idx:]
        clip = head(clip)
        clip, target = mixup.forward(clip, target)
        clip = tail(clip)
        return clip, target

    return helper

def gen_noise(num_samples: int, psd_shape_func:Callable)-> torch.Tensor:
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
    shape_signal = shape_signal / torch.sqrt(torch.mean(shape_signal**2))
    # Adjust frequency amplitudes according to noise type
    noise = white_signal * shape_signal
    return torch.fft.irfft(noise)

def gen_noise_func(f):
    """
    Given PSD shape function, returns a new function that takes in parameter N
    and generates noise Tensor of length N
    """
    return lambda N: gen_noise(N, f)

@gen_noise_func
def white_noise(_):
    """White noise PSD shape"""
    return 1

@gen_noise_func
def blue_noise(f):
    """Blue noise PSD shape"""
    return torch.sqrt(f)

@gen_noise_func
def violet_noise(f):
    """Violet noise PSD shape"""
    return f

@gen_noise_func
def brown_noise(f):
    """Brown noise PSD shape"""
    return 1/torch.where(f == 0, float('inf'), f)

@gen_noise_func
def pink_noise(f):
    """Pink noise PSD shape"""
    return 1/torch.where(f == 0, float('inf'), torch.sqrt(f))

# For some reason this class can't be printed in the repl,
# but works fine in scripts?
class SyntheticNoise(torch.nn.Module):
    """
    Attributes:
        noise_type: type of noise to add to clips
        alpha: Strength (proportion) of original audio in augmented clip
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
        augmented = self.alpha * clip + (1-self.alpha)* noise
        # Normalize noise to be between 0 and 1
        return utils.norm(augmented)


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
            q = rand(*self.q_range)
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, q)
        return clip

class BackgroundNoise(torch.nn.Module):
    """
    torch module for adding background noise to audio tensors
    Attributes:
        alpha: Strength (proportion) of original audio in augmented clip
        sample_rate: Sample rate (Hz)
        length: Length of audio clip (s)
    """
    def __init__(
            self, noise_path: Path, alpha: float, length=5
            ):
        super().__init__()
        if isinstance(noise_path, str):
            self.noise_path = Path(noise_path)
        elif isinstance(noise_path, Path):
            self.noise_path = noise_path
        else:
            raise TypeError('noise_path must be of type Path or str')
        self.alpha = alpha
        self.sample_rate = config.get_args("sample_rate")
        self.length = length

    def forward(self, clip: torch.Tensor)->torch.Tensor:
        """
        Mixes clip with noise chosen from noise_path
        Args:
            clip: Tensor of audio data

        Returns: Tensor of original clip mixed with noise
        """
        # If loading fails, skip for now
        try:
            noise_clip = self.choose_random_noise()
        except RuntimeError:
            print('Error loading noise clip, '
                  + 'background noise augmentation not performed')
            return clip
        return self.alpha*clip + (1-self.alpha)*noise_clip

    def choose_random_noise(self):
        """
        Returns: Tensor of random noise, loaded from self.noise_path
        """
        audio_extensions = (".mp3",".wav",".ogg",".flac",".opus",".sphere")
        files = list(os.listdir(self.noise_path))
        noise_clips = [f for f in files if f.endswith(audio_extensions)]
        rand_idx = utils.randint(0, len(noise_clips))
        noise_file = self.noise_path/noise_clips[rand_idx]
        clip_len = self.sample_rate*self.length
        waveform, sr = torchaudio.load(noise_file)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=self.sample_rate)
        waveform = utils.norm(waveform)
        start_idx = utils.randint(0, len(waveform))
        return waveform[start_idx, start_idx+clip_len]


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
