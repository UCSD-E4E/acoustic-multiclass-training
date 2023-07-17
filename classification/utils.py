""" 
    Stores useful functions for the classification module 
    Methods:
        print_verbose: prints the arguments if verbose is set to True

"""

from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config import get_config

def print_verbose(*args, **kwargs):
    """ 
        Prints the arguments if verbose is set to True
    """
    if("verbose" in kwargs and kwargs["verbose"]):
        del kwargs["verbose"]
        print(*args, **kwargs)

def set_seed(seed: int):
    """ Sets numpy and pytorch seeds to the CONFIG.seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def norm(s):
    """ 
    Normalizes the tensor
    """
    return s/torch.linalg.vector_norm(s, ord=2)

def randint(low, high):
    """
    Returns a random integer between [low, high)
    """
    return int(torch.randint(low, high, (1,)))

def rand(low, high):
    """
    Returns a random float between [low, high)
    """
    return (high - low) * float(torch.rand(1)[0]) + low

def get_args(*args):
    """
    Args:
        *args: Series of strings corresponding to the command line arguments
    Returns: Values of the command line arguments
    """
    CONFIG = get_config().__dict__
    return itemgetter(*args)(CONFIG)
    
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

def one_hot(t, num_classes, on_value=1., off_value=0.):
    t = t.long().view(-1, 1)
    return torch.full((t.size()[0], num_classes), off_value, device=t.device).scatter_(1, t, on_value)

def get_annotation(df: pd.DataFrame, 
        index: int,
        class_to_idx: Dict[str, Any], 
        sample_rate: int,
        target_num_samples: int,
        device,
        config) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns tuple of audio waveform and its one-hot label
    """
    #annotation = self.samples.iloc[index]
    annotation = df.iloc[index]
    file_name = annotation[config.file_name_col]

    # Turns target from integer to one hot tensor vector. I.E. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    class_name = annotation[config.manual_id_col]


    num_classes = len(set(class_to_idx.values()))
    target = one_hot(
            torch.tensor(class_to_idx[class_name]),
            num_classes)[0]
    target = target.float()

    try:
        # Get necessary variables from annotation
        annotation = df.iloc[index]
        file_name = annotation[config.file_name_col]
        frame_offset = int(annotation[config.offset_col] * sample_rate)
        num_frames = int(annotation[config.duration_col] * sample_rate)

        # Load audio
        audio = torch.load(Path(config.data_path)/file_name)
    
        if audio.shape[0] > num_frames:
            audio = audio[frame_offset:frame_offset+num_frames]
        else:
            print_verbose("SHOULD BE SMALL DELETE LATER:", audio.shape, verbose=config.verbose)

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

    #Assume audio is all mono and at target sample rate
    #assert audio.shape[0] == 1
    #assert sample_rate == self.target_sample_rate
    #audio = self.to_mono(audio) #basically reshapes to col vect

    audio = audio.to(device)
    target = target.to(device)
    return audio, target

