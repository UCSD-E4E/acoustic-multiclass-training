""" 
Contains class for applying white/pink/brown/blue/violet noise augmentation. 
The class acts as a regular torch module and can be used in a torch.nn.Sequential 
object.
"""
import torch
import numpy as np
from typing import Callable

def noise_from_PSD_func(num_samples: int, func_PSD:Callable)-> np.ndarray:
    #Reverse fourier transfrom of random array to get white noise
        white_noise = np.fft.rfft(np.random.randn(num_samples));
        shaped_signal = func_PSD(np.fft.rfftfreq(num_samples))
        # Normalize signal 
        shaped_signal = shaped_signal / np.sqrt(np.mean(shaped_signal**2))
        # Adjust frequency amplitudes according to noise type
        noise = white_noise * shaped_signal;
        return np.fft.irfft(noise);

def gen_noise_func(f):
    return lambda N: noise_from_PSD_func(N, f)

@gen_noise_func
def white_noise(f):
    return 1;

@gen_noise_func
def blue_noise(f):
    return np.sqrt(f);

@gen_noise_func
def violet_noise(f):
    return f;

@gen_noise_func
def brown_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@gen_noise_func
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

class SyntheticNoise(torch.nn.Module):
    noise_names = {'pink': pink_noise,
                   'brown': brown_noise,
                   'violet': violet_noise,
                   'blue': blue_noise,
                   'white': white_noise}
    # Potential TODO: Use SNR instead of alpha
    def __init__(
            self, noise_type: str, alpha: float, sr:int=44100, length:int=5
            ):
        self.noise_type = noise_type
        self.alpha = alpha
        self.sr = sr
        self.length = length
        self.num_samples = self.sr * self.length
        self.noise_function = self.noise_names[self.noise_type]
    def forward(self, clip: torch.Tensor)->torch.Tensor:
        noise = self.noise_function(self.num_samples)
        # Compress noise to be between 0 and 1
        # TODO: Check when to do (0,1) vs (-1, 1)
        noise = torch.tensor((noise-np.mean(noise))/(max(noise)-min(noise)))
        return self.alpha*clip + (1-self.alpha)*noise
