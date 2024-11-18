"""
File containing data augmentations implemented as torch.nn.Module
Each augmentation is initialized with only a Config object
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd #pandas: data analysis and manipulation library in python
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
#returns the inverse of each element in list as a float (1/x) unless it is index 0

def hyperbolic(seq: Iterable[int]) -> List[Tuple[float, int]]:
    """
    Takes a list of numbers and assigns them a probability
    distribution accourding to the inverse of their values
    """
    invert_seq = invert(seq)
    norm_factor = sum(invert_seq)
    probabilities = [x/norm_factor for x in invert_seq]
    return list(zip(probabilities, seq))
#takes a list of numbers (int) and returns the probability of each (for each x, inverse / sum of inverse)
#inverse--smaller numbers have higher probabilities
#WHY do we want to place more importance on smaller numbers?

def sample(distribution: List[Tuple[float, int]]) -> int:
    """
    Sample single value from distribution given by list of tuples
    """
    probabilities, values = zip(*distribution) #unzips a tuple into probabilities (x) and values (y)
    return np.random.choice(values, p = probabilities) #part of numpy library #choose a random value with differently weighted probabilities


def gen_uniform_values(n: int, min_value=0.05) -> List[float] :
    """
    Generates n values uniformly such that their sum is 1
    Args:
        n: number of values to generate, must be at least two
        min_value: Minimum possible value in list. Must be less than 1/(n-1)
    Returns: List of n values
    """
    step = 1/(n-1) #spacing between each number in interval [0,1]
    rand_points = np.arange(0, 1, step = step) #(start, stop, step)
    rand_points = [0.] + [p + utils.rand(0, step-min_value) for p in rand_points] #add randomness to each point, min of 0.05 step
    alphas = (
        [1 - rand_points[-1]] +
        [rand_points[i] - rand_points[i-1] for i in range(1, n)] #compute consecutive differences, total must be equal to 1
    )
    assert sum(alphas) <=1.00005 #check if alphas is less than 1.00005
    assert sum(alphas) >=0.99995 #check if alphas is more than 0.99995
    return alphas
#returns a list of floats that will add up to 1
#is there a fixed number of floats?

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
            df: pd.DataFrame, #pandas dataframe, files and labels? 
            class_to_idx: Dict[str, Any], #maps class labels to index values
            cfg: config.Config #mixup parameters? (e.g. 50-50 blend of cat-dog), create synthetic training examples
            ):
        super().__init__()
        self.df = df
        self.class_to_idx = class_to_idx #class to index mapping
        self.prob = cfg.mixup_p #stores probability of mixup
        self.cfg = cfg
        self.ceil_interval = cfg.mixup_ceil_interval #max number of clips to mix or highest proportion of new clips to add
        self.min_alpha = cfg.mixup_min_alpha #min proportion of mixup

        # Get probability distribution for how many clips to mix
        possible_num_clips = list(range(
                cfg.mixup_num_clips_range[0],
                cfg.mixup_num_clips_range[1] + 1)) #holds possible num of clips to mix together
        self.num_clips_distribution = hyperbolic(possible_num_clips) #randomly select probability/number of clips to mix during training

    def get_rand_clip(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        #note: tensor = holds data, in this case, audio files or labels
        """
        Get random clip from self.df
        """
        idx = utils.randint(0, len(self.df)) #get a random index in self.df
        try:
            clip, target = utils.get_annotation(
                    df = self.df,
                    index = idx, 
                    conf = self.cfg,
                    class_to_idx = self.class_to_idx) #get a random clip
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
        annotations = other_annotations + [(clip, target)] #new list with other annotations, clips, target
        clips, targets = zip(*annotations) #separate clips and targets
        mix_factors = gen_uniform_values(len(annotations), min_value = self.min_alpha) #return a list adding up to 1, min_alpha = min weight/mixing factor

        mixed_clip = sum(c * f for c, f in zip(clips, mix_factors)) #combines clip with mix factor
        mixed_target = sum(t * f for t, f in zip(targets, mix_factors)) #combines target with mix factor
        assert isinstance(mixed_target, torch.Tensor) #check mixed target is a tensor
        assert isinstance(mixed_clip, torch.Tensor) #check mixed clip is a tensor
        assert mixed_clip.shape == clip.shape #check mixed shape is same as OG shape
        assert mixed_target.shape == target.shape #check mixed target is same as OG shape
        mixed_target = utils.ceil(mixed_target, interval = self.ceil_interval) #round mixed to specific label value/interval/precision?
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
        if utils.rand(0,1) <= self.prob: #if random num bw 0 and 1 <= self.prob
            return clip, target

        num_other_clips = sample(self.num_clips_distribution) #determine how many clips to mix (rand num from list of rand numbers)
        other_annotations = [self.get_rand_clip() for _ in range(num_other_clips)] #get tuple of clip tensor and target tensor for num_other_clips (x number of ) clips
        other_annotations = list(filter(None, other_annotations)) #filter out "none"/invalid clips
        return self.mix_clips(clip, target, other_annotations) #mix clips and return


def gen_noise(num_samples: int, psd_shape_func: Callable) -> torch.Tensor:
    """
    Args:
        num_samples: length of noise Tensor to generate
        psd_shape_func: function that gives the shape of the noise's
        power spectrum distribution #control frequency distribution
        device: CUDA or CPU for processing

    Returns: noise Tensor of length num_samples
    """
    #note: PSD = power spectral density in power/Hertz; describes how the power of a signal is distributed over different frequencies
    ##(shows which frequencies contain the most energy in a signal) - different energies result in different colored noises
    #Reverse fourier transfrom of random array to get white noise
    white_signal = torch.fft.rfft(torch.rand(num_samples)) #transform random array of values to frequency domain/white noise
    # Adjust frequency amplitudes according to
    # function determining the psd shape
    shape_signal = psd_shape_func(torch.fft.rfftfreq(num_samples)) #change to different colored noises? (what's psd?)
    # Normalize signal
    shape_signal = shape_signal / torch.sqrt(torch.mean(shape_signal.float()**2)) #standard power level (why?)
    # Adjust frequency amplitudes according to noise type
    noise = white_signal * shape_signal #modify in accordance to white noise?
    return torch.fft.irfft(noise) #return noise tensor

def noise_generator(func: Callable):
    """
    Given PSD shape function, returns a new function that takes in parameter N
    and generates noise Tensor of length N
    """
    return lambda N: gen_noise(N, func) #make custom noise tensor with chosen length

@noise_generator
def white_noise(vec: torch.Tensor):
    """White noise PSD shape"""
    return torch.ones(vec.shape) #create a tensor filled with 1 with same shape as vec

@noise_generator
def blue_noise(vec: torch.Tensor):
    """Blue noise PSD shape"""
    return torch.sqrt(vec) #create tensor that's the square root as vec

@noise_generator
def violet_noise(vec: torch.Tensor):
    """Violet noise PSD shape"""
    return vec #return vec

@noise_generator
def brown_noise(vec: torch.Tensor):
    """Brown noise PSD shape"""
    return 1/torch.where(vec == 0, float('inf'), vec) #make brown noise

@noise_generator
def pink_noise(vec: torch.Tensor):
    """Pink noise PSD shape"""
    return 1/torch.where(vec == 0, float('inf'), torch.sqrt(vec)) #make pink noise

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
        self.noise_type = cfg.noise_type #choose color, also maybe psd?
        self.alpha = cfg.noise_alpha #strength of noise
        self.device = cfg.prepros_device #specify data processing device (cpu,etc.)

    def forward(self, clip: torch.Tensor)->torch.Tensor:
        """
        Args:
            clip: Tensor of audio data

        Returns: Clip mixed with noise according to noise_type and alpha
        """
        noise_function = self.noise_names[self.noise_type] #specify noise type in noise_names
        noise = noise_function(len(clip)).to(self.device) #noise tensor same len as clip
        return (1 - self.alpha) * clip + self.alpha* noise #combine OG clip and new noise tensor (alpha = amount of new noise (bigger alpha = more new))


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
        self.f_range = cfg.rand_eq_f_range #range for frequency
        self.g_range = cfg.rand_eq_g_range #range for gain, a measure of + or - amplification of a signal
        self.q_range = cfg.rand_eq_q_range #range for q factor, which "filters" frequencies (bigger, the more distinguishable)
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
        for _ in range(self.iterations): #loop for self.iterations amount of times
            frequency = utils.log_rand(*self.f_range) #generate random frequency
            gain = utils.rand(*self.g_range) #generate random gain
            q_val = utils.rand(*self.q_range) #generate random q factor
            clip = torchaudio.functional.equalizer_biquad(
                clip, self.sample_rate, frequency, gain, q_val) #apply factors to clip
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
        self.noise_path = Path(cfg.bg_noise_path) #holds path to bg noise audio files
        self.noise_path_str = cfg.bg_noise_path #bg noise path as a string
        self.alpha_range = cfg.bg_noise_alpha_range #control strength of bg noise
        self.sample_rate = cfg.sample_rate #how frequently audio is sampled
        self.length = cfg.chunk_length_s #split audio into chunks (len of clips)
        self.device = cfg.prepros_device #processing device
        self.norm = norm
        if self.noise_path_str != "" and cfg.bg_noise_p > 0.0: #check there IS bg noise and a probability to use bg noise
            files = list(os.listdir(self.noise_path)) #list all audio files in self.noise_path
            audio_extensions = (".mp3",".wav",".ogg",".flac",".opus",".sphere",".pt") #tuple of supported audio types
            self.noise_clips = [f for f in files if f.endswith(audio_extensions)] #filter to only those w supported audio types
            if len(self.noise_clips) == 0:
                raise RuntimeError("Background noise path specified, but no audio files found. " \
                                   + "Check supported format list in augmentations.py") #no audio files/incorrect path
        elif cfg.bg_noise_p!=0.0:
            raise RuntimeError("Background noise probability is non-zero, "
            + "yet no background path was specified. Please update config.yml") #no noise path named
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
        alpha = utils.rand(*self.alpha_range) #strength of audio
        if self.noise_path_str == "": #if empty noise path
            return clip
        # If loading fails, skip for now
        try:
            noise_clip = self.choose_random_noise() #tensor of random noises
        except RuntimeError as e:
            logger.warning('Error loading noise clip, background noise augmentation not performed')
            logger.error(e)
            return clip
        return (1 - alpha)*clip + alpha*noise_clip #mix clip with noise from noise_path

    def choose_random_noise(self):
        """
        Returns: Tensor of random noise, loaded from self.noise_path
        """
        rand_idx = utils.randint(0, len(self.noise_clips)) #rand index from noise clips
        noise_file = self.noise_path / self.noise_clips[rand_idx] #choose file
        clip_len = self.sample_rate * self.length #choose clip length

        if str(noise_file).endswith(".pt"): #pytorch tensor file
            waveform = torch.load(noise_file).to(self.device, dtype=torch.float32)/32767.0 #load tensor, move to device, scale to normalize amplitude
        else:
            # pryright complains that load isn't called from torchaudio. It is.
            waveform, sample_rate = torchaudio.load(noise_file, normalize=True) #pyright: ignore #load/return waveform and sample rate, scale/normalize amplitude
            waveform = waveform[0].to(self.device) #move 1st channel to device
            if sample_rate != self.sample_rate: #if loaded sample rate != required sample rate
                waveform = torchaudio.functional.resample(
                        waveform, orig_freq=sample_rate, new_freq=self.sample_rate) #resample waveform
                torch.save((waveform*32767).to(dtype=torch.int16), noise_file.with_suffix(".pt")) #convert waveform to int and save as .pt file
                os.remove(noise_file) #remove OG file
                file_name = self.noise_clips[rand_idx] #select name of a noise clip at index rand_idx
                self.noise_clips.remove(file_name) #remove OG file name
                self.noise_clips.append(str(Path(file_name).with_suffix(".pt").name)) #replace OG file name with new .pt file name
        if self.norm:
            waveform = utils.norm(waveform) #further normalize?
        start_idx = utils.randint(0, len(waveform) - clip_len) #start at random idx (make sure clip fits in waveform)
        return waveform[start_idx:start_idx+clip_len] #return segment of noise to use as background noise


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
        self.sample_rate = cfg.sample_rate #num of samples per sec
        self.cutoff = cfg.lowpass_cutoff #cutoff frequency
        self.q_val = cfg.lowpass_q_val #sharpness/strictness of cutoff

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
                                                    self.q_val) #only allow frequencies below cutoff frequency (isolate lower frequencies)

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
                                                    self.q_val) #isolate higher frequencies
