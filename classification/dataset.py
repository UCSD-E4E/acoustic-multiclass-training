from typing import Dict, List, Tuple
import torch
import torchaudio
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchaudio import transforms as audtr
import numpy as np
import os
from functools import partial
from data_aug.mixup import FastCollateMixup

from default_parser import create_parser
parser = create_parser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter



class BirdCLEFDataset(datasets.DatasetFolder):
    def __init__(self, root, loader=None, CONFIG=None, max_time=5, train=True):
        super().__init__(root, loader, extensions='wav')
        self.config = CONFIG
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

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """ Finds the classes from a directory and returns a tuple of (a list of class names, a dictionary of class names to indexes)
        """
        # modify default find_classes to ignore empty folders
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        # filter
        classes = [cls_name for cls_name in classes if len(os.listdir(os.path.join(directory, cls_name))) > 0]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.num_classes = len(classes)
        self.class_id_to_num_samples = {i: len(os.listdir(os.path.join(directory, cls_name))) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __getitem__(self, index):
        """ Returns a tuple of (image,label) for a sample with given index
        """
        
        # Load audio
        path, target = self.samples[index]
        audio, sample_rate = torchaudio.load(path)
        audio = self.to_mono(audio)
        audio = audio.to(device)
        
        # Resample
        if sample_rate != self.target_sample_rate:
            resample = audtr.Resample(sample_rate, self.target_sample_rate)
            resample.cuda(device)
            audio = resample(audio)
        
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

        # Mel spectrogram
        mel = self.mel_spectogram(audio)
        # label = torch.tensor(self.labels[index])
        
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
    return BirdCLEFDataset(root="/home/benc/code/acoustic-multiclass-training/all_10_species/", CONFIG=CONFIG)
    #train_data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/training", CONFIG=CONFIG)
    #val_data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_split_chunks_new/validation", CONFIG=CONFIG)
    #data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=CONFIG)
    #no_bird_data = BirdCLEFDataset(root="/share/acoustic_species_id/no_bird_10_000_audio_chunks", CONFIG=CONFIG)
    #data = torch.utils.data.ConcatDataset([data, no_bird_data])
    #train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    return train_data, val_data

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    CONFIG = parser.parse_args()
    CONFIG.logging = True if CONFIG.logging == 'True' else False
    # torch.manual_seed(CONFIG.seed)
    #train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    train_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs,
        #collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    )
    #val_dataloader = torch.utils.data.DataLoader(
    #    val_dataset,
    #    CONFIG.valid_batch_size,
    #    shuffle=False,
    #    num_workers=CONFIG.jobs,
    #    collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    #)
    for batch in train_dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[0].device)
        print(batch[1].device)
        break