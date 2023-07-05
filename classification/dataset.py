""" Contains methods for loading the dataset and also creates dataloaders for training and validation
    
    BirdCLEFDataset is a generic loader with a given root directory. 
    It loads the audio files and converts them to mel spectrograms.
    get_datasets returns the train and validation datasets as BirdCLEFDataset objects.
    
    If this module is run directly, it tests that the dataloader works and prints the shape of the first batch.

"""

# Standard library imports
from typing import Dict, List, Tuple
import os

# Math library imports
import pandas as pd
import numpy as np

# Torch imports
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as audtr

# Local imports
from utils import print_verbose, set_seed
from config import get_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# pylint: disable=too-many-instance-attributes
class PyhaDF_Dataset(Dataset): #datasets.DatasetFolder
    """ A dataset that loads audio files and converts them to mel spectrograms
    """
    # pylint: disable=too-many-arguments
    def __init__(self, csv_file, CONFIG=None, max_time=5, train=True, species=None, ignore_bad=True):
        super()
        if isinstance(csv_file,str):
            self.samples = pd.read_csv(csv_file, index_col=0)
        elif isinstance(csv_file,pd.DataFrame):
            self.samples = csv_file
            self.csv_file = f"data_train-{train}.csv"
        else:
            raise RuntimeError("csv_file must be a str or dataframe!")

        self.formatted_csv_file = "not yet formatted"
        self.config = CONFIG
        self.ignore_bad = ignore_bad
        self.target_sample_rate = CONFIG.sample_rate
        num_samples = CONFIG.sample_rate * max_time
        self.num_samples = num_samples
        self.mel_spectogram = audtr.MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=self.config.n_mels, 
                                        n_fft=self.config.n_fft)
        self.mel_spectogram.cuda(device)
        self.train = train
        self.freq_mask = audtr.FrequencyMasking(freq_mask_param=self.config.freq_mask_param)
        self.time_mask = audtr.TimeMasking(time_mask_param=self.config.time_mask_param)

        if species is not None:
            # pylint: disable=fixme
            #TODO FIX REPLICATION CODE
            self.classes, self.class_to_idx = species
        else:
            self.classes = self.samples[self.config.manual_id_col].unique()
            class_idx = np.arange(len(self.classes))
            self.class_to_idx = dict(zip(self.classes, class_idx))
        self.num_classes = len(self.classes)

        self.verify_audio_files()
        self.format_audio()
        #self.samples[self.config.file_path_col] = self.samples[self.config.file_path_col].apply(self.convert_file_type)
        

    def verify_audio_files(self) -> bool:
        """ Checks that all files in the dataframe exist
        """
        test_df = self.samples[self.config.file_path_col].apply(lambda path: (
            "SUCCESS" if os.path.exists(path) else path
        ))
        missing_files = test_df[test_df != "SUCCESS"].unique()
        if (missing_files.shape[0] > 0 and not self.ignore_bad):
            print(missing_files)
            raise RuntimeError("ERROR MISSING FILES, CHECK DATAFRAME")
        if self.ignore_bad:
            print("ignoring", missing_files.shape[0], "missing files")
            self.samples = self.samples[
                ~self.samples[self.config.file_path_col].isin(missing_files)
            ]
        
        print_verbose(self.samples.shape[0], "files found", verbose=self.config.verbose)

        #Run the data getting code and check to make sure preprocessing did not break code
        #poor files may contain null values, or sections of files might contain null files
        bad_files = []
        for i, (spectrogram, _) in enumerate(self):
            if spectrogram.isnan().any():
                bad_files.append(i)

        self.samples = self.samples.drop(bad_files)
        if len(bad_files) > 0 and not self.ignore_bad: 
            print("removed", len(bad_files), "corrupted annotations")
        print("Annotations count:", self.samples.shape[0])
        return True

    def get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """ Returns tuple of class list and class to index dictionary
        """
        return self.classes, self.class_to_idx
    
    def get_num_classes(self) -> int:
        """ Returns number of classes
        """
        return self.num_classes
    
    def get_csv_files(self) -> Tuple[str, str]:
        """ Returns tuple of original csv file and formatted csv file
        """
        return self.csv_file, self.formatted_csv_file
    
    def get_DF(self) -> pd.DataFrame:
        """ Returns dataframe of all annotations
        """
        return self.samples

    def format_audio(self):
        """ Formats all audio files in the list of annotations
            Saves new file paths in a file ending in formatted.csv
        """
        files = pd.DataFrame(
            self.samples[self.config.file_path_col].unique(),
            columns=["files"])
        files = files["files"].apply(self.resample_audio_file)
        self.samples = self.samples.merge(files, how="left", 
                       left_on=self.config.file_path_col,
                       right_on="IN FILE")
        
        self.samples["original_file_path"] = self.samples[self.config.file_path_col]

        if "files" in self.samples.columns:
            self.samples[self.config.file_path_col] = self.samples["files"].copy()
        if "files_y" in self.samples.columns:
            self.samples[self.config.file_path_col] = self.samples["files_y"].copy()
        
        self.formatted_csv_file = ".".join(self.csv_file.split(".")[:-1]) + "formatted.csv"
        self.samples.to_csv(self.formatted_csv_file)
        #print(self.samples[self.config.file_path_col].iloc[0], self.config.file_path_col)
        

    def resample_audio_file(self, path: str) -> pd.Series:
        """ Converts audio at path to mono and resamples to target sample rate
            Saves as new file
        """
        audio, sample_rate = torchaudio.load(path)
        changed = False

        if len(audio.shape) > 1:
            audio = self.to_mono(audio)
            changed = True
        
        # Resample
        if sample_rate != self.target_sample_rate:
            resample = audtr.Resample(sample_rate, self.target_sample_rate)
            #resample.cuda(device)
            audio = resample(audio)
            changed = True
        
        extension = path.split(".")[-1]
        new_path = path.replace(extension, "wav")

        if (new_path != path or changed):
            #output of mono is a col vector
            #torchaudio expects a waveform as row vector
            #hence the reshape
            torchaudio.save(
                new_path,
                audio.reshape([1, -1]),
                self.target_sample_rate
            )

        return pd.Series(
            {
            "IN FILE": path,    
            "files": new_path,
            }
        ).T

    def __len__(self) -> int:
        """ Returns number of annotations
        """
        return self.samples.shape[0]
        
    def get_clip(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns tuple of audio waveform and its one-hot label
        """
        annotation = self.samples.iloc[index]
        path = annotation[self.config.file_path_col]
        sample_per_sec = self.target_sample_rate
        frame_offset = int(annotation[self.config.offset_col] * sample_per_sec)
        num_frames = int(annotation[self.config.duration_col] * sample_per_sec)

        # Turns target from integer to one hot tensor vector. I.E. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        class_name = annotation[self.config.manual_id_col]

        def one_hot(x, num_classes, on_value=1., off_value=0.):
            x = x.long().view(-1, 1)
            return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)

        target = one_hot(
                torch.tensor(self.class_to_idx[class_name]),
                self.num_classes)[0]
        target = target.float()

        audio, sample_rate = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames)

        #Assume audio is all mono and at target sample rate
        assert audio.shape[0] == 1
        assert sample_rate == self.target_sample_rate
        audio = self.to_mono(audio) #basically reshapes to col vect

        # Crop if too long
        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)
        # Pad if too short
        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)

        audio = audio.to(device)
        target = target.to(device)
        return audio, target


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Takes an index and returns tuple of spectrogram image with corresponding label
        """

        audio, target = self.get_clip(index)

        # Randomly shift audio
        if self.train and torch.rand(1) < self.config.time_shift_p:
            shift = torch.randint(0, self.num_samples, (1,))
            audio = torch.roll(audio, shift, dims=1)
        # Add noise
        if self.train and torch.randn(1) < self.config.noise_p:
            noise = torch.randn_like(audio) * self.config.noise_std
            audio = audio + noise
        # Mixup
        if self.train and torch.randn(1) < self.config.mix_p:
            audio_2, target_2 = self.get_clip(np.random.randint(0, self.__len__()))
            alpha = np.random.rand() * 0.3 + 0.1
            audio = audio * alpha + audio_2 * (1 - alpha)
            target = target * alpha + target_2 * (1 - alpha)

        # Mel spectrogram
        mel = self.mel_spectogram(audio)

        # Convert to Image
        image = torch.stack([mel, mel, mel])
        
        # Normalize Image
        max_val = torch.abs(image).max() + 0.000001
        image = image / max_val
        
        # Frequency masking and time masking
        if self.train and torch.randn(1) < self.config.freq_mask_p:
            image = self.freq_mask(image)
        if self.train and torch.randn(1) < self.config.time_mask_p:
            image = self.time_mask(image)

        if image.isnan().any():
            print("ERROR IN ANNOTATION #", index)
            raise RuntimeError("NANS IN INPUT FOUND")
        return image, target
            
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
        return torch.mean(audio, axis=0)
    

    



def get_datasets(path="testformatted.csv", CONFIG=None):
    """ Returns train and validation datasets
    """
    data = pd.read_csv(path)
    train = data.sample(frac=1/2)
    valid = data[~data.index.isin(train.index)]
    return PyhaDF_Dataset(csv_file=train, CONFIG=CONFIG), PyhaDF_Dataset(csv_file=valid,train=False, CONFIG=CONFIG)

def main():
    """ Main function
    """
    torch.multiprocessing.set_start_method('spawn')
    CONFIG = get_config()
    set_seed(CONFIG.seed)
    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)

    # note: this calls __getitem__ on the dataset and discards the result
    for i, _ in enumerate(train_dataset):
        print_verbose("train entry", i,verbose=CONFIG.verbose)
    print("Loaded all training data")

    for i, _ in enumerate(val_dataset):
        print_verbose("validation entry", i,verbose=CONFIG.verbose)
    print("Loaded all validation data")

if __name__ == '__main__':
    main()
