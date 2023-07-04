import torch
import torchaudio
import torch.nn.functional as F
from torchaudio import transforms as audtr
from torch.utils.data import Dataset

from typing import Dict, List, Tuple
import os

import pandas as pd
import numpy as np

from utils import print_verbose, set_seed
from default_parser import create_parser
parser = create_parser()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PyhaDF_Dataset(Dataset):
    def __init__(self, df, csv_file="test.csv", CONFIG=None, max_time=5, train=True, species=None, ignore_bad=True):
        self.samples = df
        self.csv_file = csv_file
        self.config = CONFIG
        self.fast_verify = True
        self.formatted_csv_file = "not yet formatted"
        target_sample_rate = CONFIG.sample_rate
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples
        self.mel_spectogram = audtr.MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=self.config.n_mels, 
                                        n_fft=self.config.n_fft)
        self.mel_spectogram.cuda(device)
        self.train = train
        self.freq_mask = audtr.FrequencyMasking(freq_mask_param=self.config.freq_mask_param)
        self.time_mask = audtr.TimeMasking(time_mask_param=self.config.time_mask_param)
        self.bad_files = []

        if species is not None:
            #TODO FIX REPLICATION CODE
            self.classes, self.class_to_idx = species
        else:
            self.classes = self.samples[self.config.manual_id_col].unique()
            class_idx = np.arange(len(self.classes))
            self.class_to_idx = dict(zip(self.classes, class_idx))
            #print(self.class_to_idx)
        self.num_classes = len(self.classes)


        if((~self.samples[self.config.file_path_col]).str.contains(".pt")) != 0:
            self.serialize_data(self)

    def verify_audio(self):
        test_df = self.samples[self.config.file_path_col].apply(lambda path: (
            "SUCCESS" if os.path.exists(path) else path
        ))
        missing_files = test_df[test_df != "SUCCESS"].unique()
        print("ignoring", missing_files.shape[0], "missing files")
        self.samples = self.samples[
            ~self.samples[self.config.file_path_col].isin(missing_files)
        ]
        
    def process_audio_file(self, path):
        changed = False
        if ".pt" in path:
            return pd.Series({
                "IN FILE": path,    
                "files": path
            }).T


        try:
            audio, sample_rate = torchaudio.load(path)
        except Exception as e:
            print(path, "is bad", e)
            return pd.Series({
                "IN FILE": path,    
                "files": "bad"
            }).T
        
        if len(audio.shape) > 1:
            audio = self.to_mono(audio)
      
        # Resample
        if sample_rate != self.target_sample_rate:
            resample = audtr.Resample(sample_rate, self.target_sample_rate)
            #resample.cuda(device)
            audio = resample(audio)

        exts = "." + path.split(".")
        new_path = path.replace(exts, ".pt")
        torch.save(audio, new_path)
        return pd.Series({
                "IN FILE": path,    
                "files": new_path
            }).T


    def serialize_data(self):
        print("old size:", self.samples.shape)
        self.verify_audio()
        files = pd.DataFrame(
            self.samples[self.config.file_path_col].unique(),
            columns=["files"]
        )
        files = files[files["files"] != "bad"]
        self.samples = self.samples.merge(files, how="left", 
                       left_on=self.config.file_path_col,
                       right_on="IN FILE").dropna()
    
        print("fixed size:", self.samples.shape)
        
        self.samples["original_file_path"] = self.samples[self.config.file_path_col]

        if "files" in self.samples.columns:
            self.samples[self.config.file_path_col] = self.samples["files"].copy()
        if "files_y" in self.samples.columns:
            self.samples[self.config.file_path_col] = self.samples["files_y"].copy()
        
        self.formatted_csv_file = ".".join(self.csv_file.split(".")[:-1]) + "formatted.csv"
        self.samples.to_csv(self.formatted_csv_file)

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
        
        try:
            audio = torch.load(path)[frame_offset:frame_offset+num_frames]
        except Exception as e:
            print(e)
            print(path, index)
            raise Exception("Bad Audio")


        #print(path, "test.wav", annotation[self.config.duration_col], annotation[self.config.duration_col])

        #Assume audio is all mono and at target sample rate
        #assert audio.shape[0] == 1
        #assert sample_rate == self.target_sample_rate
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


    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index): #-> Any:
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
            self.bad_files.append(index)
            #try again with a diff annotation to avoid training breaking
            image, target = self[self.samples.sample(1).index[0]]
            
        #print(image)
        #print(target)
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
        
    def get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """ Returns tuple of class list and class to index dictionary
        """
        return self.classes, self.class_to_idx
    
    def get_num_classes(self) -> int:
        """ Returns number of classes
        """
        return self.num_classes    






























class PyhaDF_Dataset(Dataset): #datasets.DatasetFolder
    """ A dataset that loads audio files and converts them to mel spectrograms
    """
    def __init__(self, csv_file, loader=None, CONFIG=None, max_time=5, train=True, species=None, ignore_bad=True):
        super()#.__init__(root, loader, extensions='wav')
        if isinstance(csv_file,str):
            self.samples = pd.read_csv(csv_file, index_col=0)
        elif isinstance(csv_file,pd.DataFrame):
            self.samples = csv_file
            self.csv_file = f"data_train-{train}.csv"
        else:
            raise RuntimeError("csv_file must be a str or dataframe!")
        
        self.fast_verify = True
        self.formatted_csv_file = "not yet formatted"
        #print(self.samples)
        self.config = CONFIG
        self.ignore_bad = ignore_bad
        target_sample_rate = CONFIG.sample_rate
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples
        self.mel_spectogram = audtr.MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=self.config.n_mels, 
                                        n_fft=self.config.n_fft)
        self.mel_spectogram.cuda(device)
        self.train = train
        self.freq_mask = audtr.FrequencyMasking(freq_mask_param=self.config.freq_mask_param)
        self.time_mask = audtr.TimeMasking(time_mask_param=self.config.time_mask_param)

        if species is not None:
            #TODO FIX REPLICATION CODE
            self.classes, self.class_to_idx = species
        else:
            self.classes = self.samples[self.config.manual_id_col].unique()
            class_idx = np.arange(len(self.classes))
            self.class_to_idx = dict(zip(self.classes, class_idx))
            #print(self.class_to_idx)
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
        if (not self.fast_verify):
            bad_files = []
            for i in range(len(self)):
                if(i % 1000 == 0):
                    print(i)


                try:
                    spectrogram, _ = self[i]
                    if spectrogram.isnan().any():
                        bad_files.append(i)
                except Exception as e:
                    #Sam if this broke something I'm sorry
                    #If there is any error, mostly likely due to headers confusing the decoder
                    #Ignore this file
                    print_verbose(e, verbose=False)
                    bad_files.append(i)
                    #input()

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
        print(self.samples.shape)

        files = pd.DataFrame(
            self.samples[self.config.file_path_col].unique(),
            columns=["files"])
        files = files["files"].apply(self.resample_audio_file)

        files = files[files["files"] != "bad"]
        self.samples = self.samples.merge(files, how="left", 
                       left_on=self.config.file_path_col,
                       right_on="IN FILE").dropna()
        
        print(self.samples.shape)
        
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
        try:
            audio, sample_rate = torchaudio.load(path)
            changed = False
        except Exception as e:
            print(path, "is bad", e)
            return pd.Series(
                {
                "IN FILE": path,    
                "files": "bad",
                }
            ).T

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
        
        try:
            audio, sample_rate = torchaudio.load(
                path,
                frame_offset=frame_offset,
                num_frames=num_frames)
        except Exception as e:
            print(e)
            print(path, index)
            raise Exception("Bad Audio")


        #print(path, "test.wav", annotation[self.config.duration_col], annotation[self.config.duration_col])

        #Assume audio is all mono and at target sample rate
        #assert audio.shape[0] == 1
        #assert sample_rate == self.target_sample_rate
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
        #print(image)
        #print(target)
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