""" Contains methods for loading the dataset and also creates dataloaders for training and validation
    
    BirdCLEFDataset is a generic loader with a given root directory. It loads the audio files and converts them to mel spectrograms.
    get_datasets returns the train and validation datasets as BirdCLEFDataset objects.
    
    If this module is run directly, it tests that the dataloader works and prints the shape of the first batch.

"""
from typing import Dict, List, Tuple
import torch
import torchaudio
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchaudio import transforms as audtr
import numpy as np
import os
from functools import partial
from data_aug.mixup import FastCollateMixup, one_hot
from torch.utils.data import Dataset
from default_parser import create_parser
import pandas as pd
parser = create_parser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter

class PyhaDF_Dataset(Dataset): #datasets.DatasetFolder
    def __init__(self, csv_file, loader=None, CONFIG=None, max_time=5, train=True, species=None, ignore_bad=True):
        super()#.__init__(root, loader, extensions='wav')
        if (type(csv_file) is str):
            self.samples = pd.read_csv(csv_file, index_col=0)
        elif (type(csv_file) is pd.DataFrame):
            self.samples = csv_file
            self.csv_file = f"data_train-{train}.csv"
        else:
            raise RuntimeError("csv_file must be a str or dataframe!")
        
        
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
        self.collate_fn = FastCollateMixup(
            prob=self.config.mix_p,
            num_classes=self.config.num_classes,
            label_smoothing=self.config.smoothing,
            mixup_alpha=self.config.mixup_alpha,
            cutmix_alpha=self.config.cutmix_alpha,
            switch_prob=0.5
        )

        if (species is not None):
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
        

    def verify_audio_files(self):
        test_df = self.samples[self.config.file_path_col].apply(lambda path: (
            "SUCCESS" if os.path.exists(path) else path
        ))
        missing_files = test_df[test_df != "SUCCESS"].unique()
        if (missing_files.shape[0] > 0 and not self.ignore_bad):
            print(missing_files)
            raise "ERROR MISSING FILES, CHECK DATAFRAME"
            return False
        elif (self.ignore_bad):
            print("ignoring", missing_files.shape[0], "missing files")
            self.samples = self.samples[
                ~self.samples[self.config.file_path_col].isin(missing_files)
            ]
        
        print(self.samples.shape[0],"files in use")

        return True

    def get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        return self.classes, self.class_to_idx
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_csv_files(self):
        return self.csv_file, self.formatted_csv_file
    
    def get_DF(self):
        return self.samples

    def format_audio(self):
        files = pd.DataFrame(
            self.samples[self.config.file_path_col].unique(),
            columns=["files"])
        files = files["files"].apply(self.resample_audio_file)
        self.samples = self.samples.merge(files, how="left", 
                       left_on=self.config.file_path_col,
                       right_on="IN FILE")
        
        self.samples["original_file_path"] = self.samples[self.config.file_path_col]

        if ("files" in self.samples.columns):
            self.samples[self.config.file_path_col] = self.samples["files"].copy()
        if ("files_y" in self.samples.columns):
            self.samples[self.config.file_path_col] = self.samples["files_y"].copy()
        
        self.formatted_csv_file = ".".join(self.csv_file.split(".")[:-1]) + "formatted.csv"
        self.samples.to_csv(self.formatted_csv_file)
        #print(self.samples[self.config.file_path_col].iloc[0], self.config.file_path_col)
        

    def resample_audio_file(self, path):
        audio, sample_rate = torchaudio.load(path)
        changed = False

        if (len(audio.shape) > 1):
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

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        annotation = self.samples.iloc[index]
        path = annotation[self.config.file_path_col]

        #sample_per_sec was defined as total_samples_in_audio/duration
        #resampling in practice doesn't do extactly target_sample_rate
        #Its close enough so we remove that computation
        sample_per_sec = self.target_sample_rate

        #TODO requires int, does this make sense?
        frame_offset = int(annotation[self.config.offset_col] * sample_per_sec)
        num_frames = int(annotation[self.config.duration_col] * sample_per_sec)

        target = self.class_to_idx[annotation[self.config.manual_id_col]]
        target = one_hot(torch.Tensor([target]), self.num_classes)[0]
        print(target, target.dtype)
        #input()

        # print(target)
        # input()
        # #target = torch.LongTensor(target))
        # print(target)

        audio, sample_rate = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames)
  
        #print(path, "test.wav", annotation[self.config.duration_col], annotation[self.config.duration_col])
        
        #Assume audio is all mono and at target sample rate
        assert audio.shape[0] == 1
        assert sample_rate == self.target_sample_rate
        audio = self.to_mono(audio) #basically reshapes to col vect
        audio = audio.to(device)

        # Crop if too long
        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)
        # Pad if too short
        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)
        # Randomly shift audio
        if self.train and torch.rand(1) < self.config.time_shift_p:
            shift = torch.randint(0, self.num_samples, (1,))
            audio = torch.roll(audio, shift, dims=1)
        # Add noise
        if self.train and torch.randn(1) < self.config.noise_p:
            noise = torch.randn_like(audio) * self.config.noise_std
            audio = audio + noise

        #print(audio)
        # Mel spectrogram
        mel = self.mel_spectogram(audio)
        # label = torch.tensor(self.labels[index])
        
        #print(mel)

        # Convert to Image
        image = torch.stack([mel, mel, mel])
        
        # Normalize Image
        max_val = torch.abs(image).max()
        image = image / max_val
        
        # Frequency masking and time masking
        if self.train and torch.randn(1) < self.config.freq_mask_p:
            image = self.freq_mask(image)
        if self.train and torch.randn(1) < self.config.time_mask_p:
            image = self.time_mask(image)

        #print(image)
        #print(target)
        return image, target
            
    def pad_audio(self, audio):
        """Fills the last dimension of the input audio with zeroes until it is num_samples long
        """
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio
        
    def crop_audio(self, audio):
        """Cuts audio to num_samples long
        """
        return audio[:self.num_samples]
        
    def to_mono(self, audio):
        return torch.mean(audio, axis=0)
    

    



def get_datasets(path="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=None):

    data = pd.read_csv("testformatted.csv")
    train = data.sample(frac=1/2)
    valid = data[~data.index.isin(train.index)]
    return PyhaDF_Dataset(csv_file=train, CONFIG=CONFIG), PyhaDF_Dataset(csv_file=valid,train=False, CONFIG=CONFIG)
    #data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=CONFIG)
    #no_bird_data = BirdCLEFDataset(root="/share/acoustic_species_id/no_bird_10_000_audio_chunks", CONFIG=CONFIG)
    #data = torch.utils.data.ConcatDataset([data, no_bird_data])
    #train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    #return train_data, val_data

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    CONFIG = parser.parse_args()
    CONFIG.logging = True if CONFIG.logging == 'True' else False
    # torch.manual_seed(CONFIG.seed)
    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    print(train_dataset.get_classes()[1])
    print(train_dataset.__getitem__(0))
    input()
    #train_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        1,
        shuffle=True,
        num_workers=CONFIG.jobs,
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     CONFIG.valid_batch_size,
    #     shuffle=False,
    #     num_workers=CONFIG.jobs,
    #     collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    # )

    for i in range(train_dataset.__len__()):
        print("entry", i)
        train_dataset.__getitem__(i)
        input()

    # print("started running batches")
    # for batch in train_dataloader:
    #     print("successfully loaded batch")
    # print("end of code")
