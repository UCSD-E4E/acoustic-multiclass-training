from typing import Dict, List, Tuple
import torch
import torchaudio
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchaudio import transforms as audtr
import numpy as np
import argparse
import os
from functools import partial
import librosa

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=264, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=32, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=32, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)
parser.add_argument('-l', '--logging', default='True', type=str)
parser.add_argument('-lf', '--logging_freq', default=20, type=int)
parser.add_argument('-vf', '--valid_freq', default=2000, type=int)
parser.add_argument('-mch', '--model_checkpoint', default=None, type=str)
parser.add_argument('-md', '--map_debug', action='store_true')
parser.add_argument('-p', '--p', default=0, type=float, help='p for mixup')


#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter

class BirdCLEFDataset(datasets.DatasetFolder):
    def __init__(self, root, loader=None, CONFIG=None, max_time=5):
        super().__init__(root, loader, extensions=['wav'])
        self.config = CONFIG
        target_sample_rate = CONFIG.sample_rate
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples
        self.max_time = max_time

        #TODO: ADD CASES TO HANDLE THIS
        #self.chunk_samples()


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
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
        path, cl = self.samples[index]
        audio, sample_rate = torchaudio.load(path)
        audio = self.to_mono(audio)
        
        if sample_rate != self.target_sample_rate:
            resample = audtr.Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)
        
        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)
            
        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)
            
        mel_spectogram = audtr.MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=self.config.n_mels, 
                                        n_fft=self.config.n_fft)
        mel = mel_spectogram(audio)
        # label = torch.tensor(self.labels[index])
        
        # Convert to Image
        image = torch.stack([mel, mel, mel])
        
        # Normalize Image
        max_val = torch.abs(image).max()
        image = image / max_val
        # one hot target
        target = torch.zeros(self.num_classes)
        target[cl] = 1
        return image, target
            
    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio
        
    def crop_audio(self, audio):
        audio  = self.random_chunk(audio)
        return audio #audio[:self.num_samples]
        
    def to_mono(self, audio):
        return torch.mean(audio, axis=0)
    
    # def chunk_samples(self):
    #     new_samples = []
    #     for path, cl in self.samples:
    #         duration = librosa.get_duration(path=path)
    #         chunks = int(duration // self.max_time)
            
    #         if (chunks == 0):
    #             new_samples.append((path, cl, 0, duration))
            
    #         for i in range(chunks):
    #             new_samples.append((path, cl, duration*i, duration*(i+1)))
        
    #     print(new_samples)


    def random_chunk(self, audio):
        start = np.random.randint(audio.shape[0] - self.num_samples)
        audio = audio[start:start + self.num_samples]
        return audio

        
    
    @staticmethod
    def collate(batch, p=0.3):
        # with probability p, randomly combine two samples from different classes
        # with probability 1-p, keep the samples as they are
        # this is to help the model learn to distinguish between classes
        # since the dataset is imbalanced
        new_batch = []
        for i in range(0, len(batch), 2):
            if np.random.rand() < p:
                new_batch.append((batch[i][0] + batch[i+1][0], torch.clamp(batch[i][1] + batch[i+1][1], max=1)))
            else:
                new_batch.append((batch[i][0], batch[i][1]))
                new_batch.append((batch[i+1][0], batch[i+1][1]))
        return torch.utils.data.default_collate(new_batch)




def get_datasets(path="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=None):
    train_data = BirdCLEFDataset(root="../acoustic-species-classification/BirdCLEF2023_split_chunks/training", CONFIG=CONFIG)
    val_data = BirdCLEFDataset(root="../acoustic-species-classification/BirdCLEF2023_split_chunks/validation", CONFIG=CONFIG)
    #data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=CONFIG)
    #no_bird_data = BirdCLEFDataset(root="/share/acoustic_species_id/no_bird_10_000_audio_chunks", CONFIG=CONFIG)
    #data = torch.utils.data.ConcatDataset([data, no_bird_data])
    #train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    return train_data, val_data

if __name__ == '__main__':
    CONFIG = parser.parse_args()
    CONFIG.logging = True if CONFIG.logging == 'True' else False
    train_data = BirdCLEFDataset(root="../acoustic-species-classification/BirdCLEF2023_split_chunks", CONFIG=CONFIG)
    print(train_data.__getitem__(0))
    print(train_data.__getitem__(0)[0].shape)
    
    
    
    
    
    
    
    
    
    
    
    # torch.manual_seed(CONFIG.seed)
    # train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     CONFIG.train_batch_size,
    #     shuffle=True,
    #     num_workers=CONFIG.jobs,
    #     collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     CONFIG.valid_batch_size,
    #     shuffle=False,
    #     num_workers=CONFIG.jobs,
    #     collate_fn=partial(BirdCLEFDataset.collate, p=CONFIG.p)
    # )
    # for batch in train_dataloader:
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     break