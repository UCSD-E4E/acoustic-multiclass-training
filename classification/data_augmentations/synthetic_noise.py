""" 
Contains class for applying white/pink/brown/blue/violet noise augmentation. 
The class acts as a regular torch module and can be used in a torch.nn.Sequential 
object.
"""
import torch
import numpy as np
from typing import Callable

def gen_noise(num_samples: int, psd_shape_func:Callable)-> np.ndarray:
    #Reverse fourier transfrom of random array to get white noise
    white_noise = np.fft.rfft(np.random.randn(num_samples));
    # Adjust frequency amplitudes according to 
    # function determining the psd shape
    shape_signal = psd_shape_func(np.fft.rfftfreq(num_samples))
    # Normalize signal 
    shape_signal = shape_signal / np.sqrt(np.mean(shape_signal**2))
    # Adjust frequency amplitudes according to noise type
    noise = white_noise * shape_signal;
    return np.fft.irfft(noise);

def gen_noise_func(f):
    return lambda N: gen_noise(N, f)

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

#Norms signal to be in range [0,1]
def norm(s):
    return s/max(s)

def mix_audio(signal, noise, snr):
    if len(signal) != len(noise):
        raise ValueError('Signal and noise must have same length')
    # To avoid overflow when squaring
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)
    
    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise 
    # to achieve the given SNR
    gain = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    
    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + gain**2))
    b = np.sqrt(gain**2 / (1 + gain**2))
    # mix the signals
    return a * signal + b * noise

# For some reason this class can't be printed in the repl, 
# but works fine in scripts?
class SyntheticNoise(torch.nn.Module):
    noise_names = {'pink': pink_noise,
                   'brown': brown_noise,
                   'violet': violet_noise,
                   'blue': blue_noise,
                   'white': white_noise}
    def __init__(self, noise_type: str, snr: float):
        self.noise_type = noise_type
        self.snr = snr
        self.noise_function = self.noise_names[self.noise_type]
    def forward(self, clip: torch.Tensor)->torch.Tensor:
        noise = self.noise_function(len(clip))
        augmented = mix_audio(clip, noise, self.snr)
        # Compress noise to be between 0 and 1
        return torch.tensor(augmented)/max(augmented)
