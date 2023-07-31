""" Stores useful functions for the pyha_analyzer module """

from pathlib import Path
from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyha_analyzer import config

cfg = config.cfg

def set_seed(seed: int):
    """ Sets numpy and pytorch seeds to the CONFIG.seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def norm(tensor: torch.Tensor):
    """ 
    Normalizes the tensor
    """
    return tensor/torch.linalg.vector_norm(tensor, ord=2)

def randint(low: int, high: int) -> int:
    """
    Returns a random integer between [low, high)
    """
    return int(torch.randint(low, high, (1,)))

def rand(low: float , high: float) -> float:
    """
    Returns a random float between [low, high)
    """
    return (high - low) * float(torch.rand(1)[0]) + low

def log_rand(low: float, high: float) -> float:
    """
    Returns a random float between [low, high), with
    a log uniform distribution
    """
    low_exp = math.log(low)
    high_exp = math.log(high)
    rand_exp = rand(low_exp, high_exp)
    return math.exp(rand_exp)

def pad_audio(audio: torch.Tensor, num_samples:int) -> torch.Tensor:
    """Fills the last dimension of the input audio with zeroes until it is num_samples long
    """
    pad_length = num_samples - audio.shape[0]
    last_dim_padding = (0, pad_length)
    audio = F.pad(audio, last_dim_padding)
    return audio

def crop_audio(audio: torch.Tensor, num_samples:int) -> torch.Tensor:
    """Cuts audio to num_samples long
    """
    return audio[:num_samples]

def to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Converts audio to mono
    """
    return torch.mean(audio, dim=0)

def one_hot(tensor, num_classes, on_value=1., off_value=0.):
    """Return one hot tensor of length num_classes
    """
    tensor = tensor.long().view(-1, 1)
    return torch.full((tensor.size()[0], num_classes), off_value, device=tensor.device) \
                .scatter_(1, tensor, on_value)
def rand_offset():
    """
    Return a random offset in samples
    """
    max_offset = int(cfg.max_offset * cfg.sample_rate)
    if max_offset == 0:
        return 0
    return randint(-max_offset, max_offset)

#pylint: disable-next = too-many-arguments
def get_annotation(
        df: pd.DataFrame, 
        index: int,
        class_to_idx: Dict[str, Any], 
        offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns tuple of audio waveform and its one-hot label
    """
    assert isinstance(index, int)
    sample_rate = cfg.sample_rate
    target_num_samples = cfg.sample_rate * cfg.chunk_length_s
    annotation = df.iloc[index]
    file_name = annotation[cfg.file_name_col]
    num_classes = len(set(class_to_idx.values()))

    # Turns target from integer to one hot tensor vector. I.E. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    class_name = annotation[cfg.manual_id_col]
    if isinstance(class_name, dict):
        target = torch.zeros(num_classes)
        for name, alpha in class_name.items():
            target[class_to_idx[name]] = alpha
    else:
        target = one_hot(
                torch.tensor(class_to_idx[class_name]),
                num_classes)[0]
    target = target.float()

    try:
        # Get necessary variables from annotation
        annotation = df.iloc[index]
        file_name = annotation[cfg.file_name_col]
        frame_offset = int(annotation[cfg.offset_col] * sample_rate)
        if offset:
            frame_offset += rand_offset()
        num_frames = int(annotation[cfg.duration_col] * sample_rate)

        # Load audio
        audio = torch.load(Path(cfg.data_path)/file_name)
    
        if audio.shape[0] > num_frames:
            audio = audio[frame_offset:frame_offset+num_frames]

        # Crop if too long
        if audio.shape[0] > target_num_samples:
            audio = crop_audio(audio, target_num_samples)
        # Pad if too short
        if audio.shape[0] < target_num_samples:
            audio = pad_audio(audio, target_num_samples)
    except Exception as e:
        print(e)
        print(file_name, index)
        raise RuntimeError("Bad Audio") from e

    audio = audio.to(cfg.prepros_device)
    target = target.to(cfg.prepros_device)
    return audio, target
