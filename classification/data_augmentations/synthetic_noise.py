""" 
Contains class for applying white/pink/brown/blue/violet noise augmentation. 
The class acts as a regular torch module and can be used in a torch.nn.Sequential 
object.
"""
from typing import Callable
import torch
import numpy as np

def gen_noise(num_samples: int, psd_shape_func:Callable)-> torch.Tensor:
    """
    Args:
        num_samples: length of noise Tensor to generate
        psd_shape_func: function that gives the shape of the noise's 
        power spectrum distribution

    Returns: noise Tensor of length num_samples
    """
    #Reverse fourier transfrom of random array to get white noise
    white_signal = np.fft.rfft(np.random.randn(num_samples))
    # Adjust frequency amplitudes according to 
    # function determining the psd shape
    shape_signal = psd_shape_func(np.fft.rfftfreq(num_samples))
    # Normalize signal 
    shape_signal = shape_signal / np.sqrt(np.mean(shape_signal**2))
    # Adjust frequency amplitudes according to noise type
    noise = white_signal * shape_signal
    return torch.Tensor(np.fft.irfft(noise))

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
    return np.sqrt(f)

@gen_noise_func
def violet_noise(f):
    """Violet noise PSD shape"""
    return f

@gen_noise_func
def brown_noise(f):
    """Brown noise PSD shape"""
    return 1/np.where(f == 0, float('inf'), f)

@gen_noise_func
def pink_noise(f):
    """Pink noise PSD shape"""
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

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
        noise = torch.Tensor(noise_function(len(clip)))
        augmented = self.alpha * clip + (1-self.alpha)* noise
        # Normalize noise to be between 0 and 1
        return augmented/torch.max(augmented)
