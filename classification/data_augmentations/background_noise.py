""" 
Contains class for applying background noise augmentation. The class acts as
a regular torch module and can be used in a torch.nn.Sequential object.
"""
import os
from pathlib import Path
import torch
import torchaudio

class BackgroundNoise(torch.nn.Module):
    """
    torch module for adding background noise to audio tensors
    Attributes: 
        alpha: Strength (proportion) of original audio in augmented clip
        sample_rate: Sample rate (Hz)
        length: Length of audio clip (s)
    """
    def __init__(
            self, noise_path: Path, alpha: float, sample_rate=44100, length=5
            ):
        super().__init__()
        if isinstance(noise_path, str):
            self.noise_path = Path(noise_path)
        elif isinstance(noise_path, Path):
            self.noise_path = noise_path
        else:
            raise TypeError('noise_path must be of type Path or str')
        self.alpha = alpha
        self.sample_rate = sample_rate
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
        rand_idx = torch.randint(len(noise_clips), (1,))[0]
        noise_file = self.noise_path/noise_clips[rand_idx]
        clip_len = self.sample_rate*self.length
        waveform, sr = torchaudio.load(noise_file)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=self.sample_rate)
        waveform = waveform/torch.max(waveform)
        start_idx = torch.randint(len(waveform)-clip_len, (1,))[0]
        return waveform[start_idx, start_idx+clip_len]
