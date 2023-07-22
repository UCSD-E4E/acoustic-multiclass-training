"""
    This file contains methods that allow the visualization of
    different data augmentations.
"""
from typing import Callable, List, Tuple

import config
import numpy as np
import torch
from augmentations import (BackgroundNoise, LowpassFilter, Mixup, 
                           RandomEQ, SyntheticNoise)
from dataset import PyhaDFDataset, get_datasets
from matplotlib import cm
from matplotlib import pyplot as plt
from utils import get_annotation

cfg = config.cfg


# PARAMETERS
DEFAULT_CONF = {
    "synth_colors": ["white","pink","brown","violet","blue"],
    "synth_noise_intensity":  0.3,
    "lowpass_cutoff": 1000,
    "lowpass_q_val": 0.707,
    "eq_f_range": (100, 6000),
    "eq_g_range": (-8, 8),
    "eq_q_range": (1, 9),
    "eq_num_applications": 1,
    "background_intensity": 0.8,
    "background_norm": False,
    "mixup_alpha_range": (0.1,0.4),
}

def sigmoid(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))
def linear(x):
    """ Linear function """
    return x
def tanh(x):
    """ Tanh function """
    return np.tanh(x)
def log1p(x):
    """ Log1p function """
    return np.log1p(x)
DEFAULT_NORMS = {
    "use_normal_dist": True,
    "normalization": log1p,
    "min_clip": -3,
    "max_clip": 3,
}

N_SAMPLES = 3

def plot_audio(audio_list: List[torch.Tensor]):
    """Plots list of audio waveforms (not currently used)"""
    plt.figure(figsize=(10,len(audio_list)))
    for i, audio in enumerate(audio_list):
        plt.subplot(len(audio_list),1,i+1)
        plt.plot(audio.to("cpu").numpy())
    plt.show()

def get_audio(dataset: PyhaDFDataset, num_samples: int=3):
    """ Returns an array of audio waveforms and an array of one-hot labels """
    return [(
        get_annotation(
            dataset.samples,
            np.random.randint(len(dataset)),
            dataset.class_to_idx,
            "cuda")[0]
    )
            for _ in range(num_samples)]

N_AUGS = 9
def get_augs(dataset: PyhaDFDataset, noise_types: list, _cfg) -> Tuple[List[Callable],List[str]]:
    """ Returns a list of augmentations that can be applied
    Each element is a tuple of the form (aug, name)
    Each augmentation is a Callable that takes in a waveform and returns a waveform
    """
    out = []
    names = []
    for col in noise_types:
        out.append(SyntheticNoise(_cfg).forward)
        names.append("Synthetic " + col +" noise")
    out.append(LowpassFilter(_cfg).forward)
    names.append("Lowpass filter")
    out.append(RandomEQ(_cfg).forward)
    names.append("Random EQ")
    out.append(BackgroundNoise(_cfg).forward)
    names.append("Background noise")
    mixup = Mixup(df = dataset.samples,
                  class_to_idx = dataset.class_to_idx,
                  cfg = _cfg)
    out.append(lambda x: mixup(x,torch.zeros(dataset.num_classes).to(x.device))[0])
    names.append("Mixup")
    return out, names

def apply_augs(audio: torch.Tensor, augs: List[Callable]) -> List[torch.Tensor]:
    """ Apply all augmetations
    Return a tuple of a list of augmented audio and a list of labels
    """ 
    return [aug(audio) for aug in augs]


def normalize(data: torch.Tensor, norms) -> np.ndarray:
    """ Take in a mel spectrogram and return a normalized version """
    if norms["use_normal_dist"]:
        data = (data-torch.mean(data))/torch.std(data)
    data_np = norms["normalization"](data.to("cpu").numpy())
    data_np = np.clip(data_np,norms["min_clip"],norms["max_clip"])
    return data_np

def get_min_max(mel_list: List[np.ndarray]) -> Tuple[float,float]:
    """ Get the min and max values of a list of mel spectrograms """
    vmax = 0
    vmin = 0
    for mel in mel_list:
        vmax = max(vmax, np.max(mel))
        vmin = min(vmin, np.min(mel))
    return vmin, vmax

def plot(mels: List[Tuple[np.ndarray,str,Tuple[int,int]]],
         vmin:float, vmax:float,  norms=None) -> None:
    """ Plots a list of mel spectrograms 
        Arguments:
            mels: List of tuples of the form (mel, title, (x,y))
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            norms: List of normalization parameters (for display)
    """
    n_samples = len(mels)//N_AUGS
    # Create subplots
    fig, axes = plt.subplots(N_AUGS,n_samples, figsize=(12,40))
    # Plot each mel
    for (mel, title, (x,y)) in mels:
        plt.subplot(N_AUGS,n_samples,y*n_samples+x+1)
        img = axes[y][x].imshow(mel, cmap="viridis", origin="lower", clim=(vmin,vmax))
        axes[y][x].set_title(title)
    # Create colorbar
    cmap = cm.ScalarMappable(cmap="viridis")
    cmap.set_array([])
    plt.colorbar(img, ax=axes, location="top") # pyright: ignore
    # Bottom text
    fig.text(0.5,0.05, str(norms))
    plt.show()

def run_test(n_samples,norms, noise_types): 
    """ Main function """
    train_ds, _ = get_datasets()
    # Get audio
    audio = get_audio(train_ds, n_samples)
    # Get augs
    augs, names = get_augs(train_ds, noise_types, cfg)
    # Apply augs
    augmented_audio = [apply_augs(audio[i],augs) for i in range(n_samples)]
    # Unpack to list of (audio, name, (x,y))
    audio_data = []
    for aug in range(len(augs)):
        for sample in range(n_samples):
            audio_data.append((augmented_audio[sample][aug], names[aug], (sample,aug)))
    # Get mels
    mels = [(train_ds.to_image(aud)[0], label, pos) for (aud, label, pos) in audio_data]
    # Normalize mels
    mels_norm = [(normalize(mel,norms), label, pos) for (mel, label, pos) in mels]
    # Get min and max value
    vmin, vmax = get_min_max([mel for (mel,_,_) in mels_norm])

    plot(mels_norm, vmin, vmax, norms)

# if __name__ == "__main__":
#     run_test(3,DEFAULT_NORMS)
