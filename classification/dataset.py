""" Contains methods for loading the dataset and creates dataloaders for training and validation

    BirdCLEFDataset is a generic loader with a given root directory.
    It loads the audio files and converts them to mel spectrograms.
    get_datasets returns the train and validation datasets as BirdCLEFDataset objects.

    If this module is run directly, it tests that the dataloader works

"""
import os
from typing import Dict, List, Tuple

import config
import numpy as np
# Math library imports
# Math library imports
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from augmentations import Mixup, add_mixup
from config import get_config
from torch.utils.data import Dataset
from torchaudio import transforms as audtr
from tqdm import tqdm
from utils import print_verbose, set_seed

cfg = config.cfg

tqdm.pandas()
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# pylint: disable=too-many-instance-attributes
class PyhaDFDataset(Dataset):
    """
        Dataset designed to work with pyha output
        Save unchunked data
    """

    # df, csv_file, train, and species decided outside of config, so those cannot be added in there
    # pylint: disable-next=too-many-arguments
    def __init__(self, df, csv_file="test.csv", train=True, species=None):
        self.samples = df[~(df[cfg.file_name_col].isnull())]
        self.csv_file = csv_file
        self.formatted_csv_file = "not yet formatted"
        self.target_sample_rate = cfg.sample_rate
        num_samples = self.target_sample_rate * cfg.max_time
        self.num_samples = num_samples
        self.train = train


        self.mel_spectogram = audtr.MelSpectrogram(sample_rate=self.target_sample_rate,
                                        n_mels=cfg.n_mels,
                                        n_fft=cfg.n_fft)
        self.mel_spectogram.to(device) #was cuda (?)
        self.freq_mask = audtr.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = audtr.TimeMasking(time_mask_param=cfg.time_mask_param)
        self.transforms = None
        self.mixup = None

        # List data directory and confirm it exists
        if not os.path.exists(cfg.data_path):
            raise FileNotFoundError("Data path does not exist")
        self.data_dir = set(os.listdir(cfg.data_path))
        
        #Log bad files
        self.bad_files = []

        #Preprocessing start
        if species is not None:
            self.classes, self.class_to_idx = species
        else:
            self.classes = self.samples[cfg.manual_id_col].unique()
            class_idx = np.arange(len(self.classes))
            self.class_to_idx = dict(zip(self.classes, class_idx))

        self.num_classes = len(self.classes)
        self.serialize_data()

    def verify_audio(self):
        """
        Checks to make sure files exist that are referenced in input df
        """
        missing_files = pd.Series(self.samples[cfg.file_name_col].unique()) \
            .progress_apply(lambda file: "good" if file in self.data_dir else file)
        missing_files = missing_files[missing_files != "good"].unique()
        print("ignoring", missing_files.shape[0], "missing files")
        self.samples = self.samples[
            ~self.samples[cfg.file_name_col].isin(missing_files)
        ]

    def process_audio_file(self, file_name):
        """
        Save waveform of audio file as a tensor and save that tensor to .pt
        """

        exts = "." + file_name.split(".")[-1]
        new_name = file_name.replace(exts, ".pt")
        if new_name in self.data_dir:
            #ASSUME WE HAVE ALREADY PREPROCESSED THIS CORRECTLY
            return pd.Series({
                "FILE NAME": file_name,
                "files": new_name
            }).T


        try:
            # old error: "load" is not a known member of module "torchaudio"
            # Load is a known member of torchaudio:
            # https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data
            audio, sample_rate = torchaudio.load(       #pyright: ignore [reportGeneralTypeIssues ]
                os.path.join(cfg.data_path, file_name)
            ) 

            if len(audio.shape) > 1:
                audio = self.to_mono(audio)

            # Resample
            if sample_rate != self.target_sample_rate:
                resample = audtr.Resample(sample_rate, self.target_sample_rate)
                #resample.cuda(DEVICE)
                audio = resample(audio)

            torch.save(audio, os.path.join(cfg.data_path,new_name))
            self.data_dir.add(new_name)
        # IO is messy, I want any file that could be problematic
        # removed from training so it isn't stopped after hours of time
        # Hence broad exception
        # pylint: disable-next=W0718
        except Exception as e:
            print_verbose(file_name, "is bad", e, verbose=cfg.verbose)
            return pd.Series({
                "FILE NAME": file_name,    
                "files": "bad"
            }).T


        return pd.Series({
                "FILE NAME": file_name,    
                "files": new_name
            }).T


    def serialize_data(self):
        """
        For each file, check to see if the file is already a presaved tensor
        If the files is not a presaved tensor and is an audio file, convert to tensor to make
        Future training faster
        """
        self.verify_audio()
        files = pd.DataFrame( self.samples[cfg.file_name_col].unique(),
            columns=["files"]
        )
        files = files["files"].progress_apply(self.process_audio_file)

        print(files.shape, flush=True)

        num_files = files.shape[0]
        if num_files == 0:
            raise FileNotFoundError("There were no valid filepaths found, check csv")

        files = files[files["files"] != "bad"]
        self.samples = self.samples.merge(files, how="left", 
                       left_on=cfg.file_name_col,
                       right_on="FILE NAME").dropna()
    
        print_verbose("Serialized form, fixed size:", self.samples.shape, verbose=cfg.verbose)

        if "files" in self.samples.columns:
            self.samples[cfg.file_name_col] = self.samples["files"].copy()
        if "files_y" in self.samples.columns:
            self.samples[cfg.file_name_col] = self.samples["files_y"].copy()
        
        self.samples["original_file_path"] = self.samples[cfg.file_name_col]

        self.formatted_csv_file = ".".join(self.csv_file.split(".")[:-1]) + "formatted.csv"
        self.samples.to_csv(self.formatted_csv_file)

    def get_annotation(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns tuple of audio waveform and its one-hot label
        """
        annotation = self.samples.iloc[index]
        file_name = annotation[cfg.file_name_col]

        # Turns target from integer to one hot tensor vector. I.E. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        class_name = annotation[cfg.manual_id_col]

        def one_hot(x, num_classes, on_value=1., off_value=0.):
            x = x.long().view(-1, 1)
            return torch.full((x.size()[0], num_classes), off_value, device=x.device) \
                .scatter_(1, x, on_value)

        target = one_hot(
                torch.tensor(self.class_to_idx[class_name]),
                self.num_classes)[0]
        target = target.float()

        try:
            # Get necessary variables from annotation
            annotation = self.samples.iloc[index]
            file_name = annotation[cfg.file_name_col]
            sample_per_sec = self.target_sample_rate
            frame_offset = int(annotation[cfg.offset_col] * sample_per_sec)
            num_frames = int(annotation[cfg.duration_col] * sample_per_sec)

            # Load audio
            audio = torch.load(os.path.join(cfg.data_path,file_name))
        
            if audio.shape[0] > num_frames:
                audio = audio[frame_offset:frame_offset+num_frames]
            else:
                print_verbose("SHOULD BE SMALL DELETE LATER:", audio.shape, verbose=cfg.verbose)

            # Crop if too long
            if audio.shape[0] > self.num_samples:
                audio = self.crop_audio(audio)
            # Pad if too short
            if audio.shape[0] < self.num_samples:
                audio = self.pad_audio(audio)
        except Exception as exc:
            print(exc)
            print(file_name, index)
            raise RuntimeError("Bad Audio") from exc

        #Assume audio is all mono and at target sample rate
        #assert audio.shape[0] == 1
        #assert sample_rate == self.target_sample_rate
        #audio = self.to_mono(audio) #basically reshapes to col vect

        audio = audio.to(DEVICE)
        target = target.to(DEVICE)
        return audio, target

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index): #-> Any:
        """ Takes an index and returns tuple of spectrogram image with corresponding label
        """

        audio, target = self.get_annotation(index)
        
        # Randomly shift audio
        if self.train and torch.rand(1) < cfg.time_shift_p:
            shift = int(torch.randint(0, self.num_samples, (1,)))
            audio = torch.roll(audio, shift, dims=1)
        
        if self.transforms is not None and self.mixup is not None:
            mixup_idx = 0
            audio, target = add_mixup(audio, 
                                     target, 
                                      self.mixup, 
                                      self.transforms, 
                                      mixup_idx)
        elif  self.transforms is not None:
            audio = self.transforms(audio)

        
        # Add noise
        #if self.train and torch.randn(1) < cfg.noise_p:
        #    noise = torch.randn_like(audio) * cfg.noise_std
        #    audio = audio + noise
        # Mixup
        #if self.train and torch.randn(1) < cfg.mix_p:
        #    audio_2, target_2 = self.get_annotation(np.random.randint(0, self.__len__()))
        #    alpha = np.random.rand() * 0.3 + 0.1
        #    audio = audio * alpha + audio_2 * (1 - alpha)
        #    target = target * alpha + target_2 * (1 - alpha)

        # Mel spectrogram
        mel = self.mel_spectogram(audio)

        # Convert to Image
        image = torch.stack([mel, mel, mel])

        # Normalize Image
        max_val = torch.abs(image).max() + 0.000001
        image = image / max_val

        # Frequency masking and time masking
        if self.train and torch.randn(1) < cfg.freq_mask_p:
            image = self.freq_mask(image)
        if self.train and torch.randn(1) < cfg.time_mask_p:
            image = self.time_mask(image)

        if image.isnan().any():
            print("ERROR IN ANNOTATION #", index)
            self.bad_files.append(index)
            #try again with a diff annotation to avoid training breaking
            image, target = self[self.samples.sample(1).index[0]]

        return image, target

    def set_transforms(self, transforms):
        """ Sets the transforms for the dataset
        """
        self.transforms = transforms
    def set_mixup(self, mixup):
        """ Sets the mixup object for the dataset
        """
        self.mixup = mixup

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Fills the last dimension of the input audio with zeroes until it is num_samples long
        """
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio

    def crop_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Cuts audio to num_samples long
        """
        return audio[:self.num_samples]

    def to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """ Converts audio to mono by averaging the channels
        """
        return torch.mean(audio, dim=0)

    def get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """ Returns tuple of class list and class to index dictionary
        """
        return self.classes, self.class_to_idx

    def get_num_classes(self) -> int:
        """ Returns number of classes
        """
        return self.num_classes


def get_datasets(transforms = None):
    """ Returns train and validation datasets, does random sampling for train/valid split, adds transforms to dataset
    """


    train_p = cfg.train_test_split
    path = cfg.dataframe_csv
    # Load the dataset
    data = pd.read_csv(path, usecols = [
        cfg.file_name_col,
        cfg.manual_id_col,
        cfg.offset_col,
        cfg.duration_col
    ], dtype={
        cfg.file_name_col: str,
        cfg.manual_id_col: str,
        cfg.offset_col: float,
        cfg.duration_col: float})
    
    #for each species, get a random sample of files for train/valid split
    train_files = data.groupby(cfg.manual_id_col, as_index=False).apply(
        lambda x: pd.Series(x[cfg.file_name_col].unique()).sample(frac=train_p)
    )
    train = data[data[cfg.file_name_col].isin(train_files)]

    valid = data[~data.index.isin(train.index)]

    train_ds = PyhaDF_Dataset(train, csv_file="train.csv")
    species = train_ds.get_classes()

    mixup_ds = PyhaDF_Dataset(train, csv_file="mixup.csv",train=False)
    mixup = Mixup(mixup_ds)

    if transforms is not None:
        train_ds.set_transforms(transforms)
        train_ds.set_mixup(mixup)

    valid_ds = PyhaDF_Dataset(valid, csv_file="valid.csv",train=False, species=species)
    return train_ds, valid_ds

def main():
    """
    testing function.
    """
    torch.multiprocessing.set_start_method('spawn')
    set_seed(cfg.seed)
    get_datasets()

if __name__ == '__main__':
    main()
