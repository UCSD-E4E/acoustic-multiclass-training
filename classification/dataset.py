import torch
import torchaudio
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchaudio import transforms as audtr


#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter

class BirdCLEFDataset(datasets.DatasetFolder):
    def __init__(self, root, loader=None, CONFIG=None, max_time=5):
        super().__init__(root, loader, extensions='wav')
        self.config = CONFIG
        target_sample_rate = CONFIG.sample_rate
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples

    
    def __getitem__(self, index):
        path, target = self.samples[index]
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
        
        return image, target
            
    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio
        
    def crop_audio(self, audio):
        return audio[:self.num_samples]
        
    def to_mono(self, audio):
        return torch.mean(audio, axis=0)


def get_datasets(path="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=None):
    data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks", CONFIG=CONFIG)
    no_bird_data = BirdCLEFDataset(root="/share/acoustic_species_id/no_bird_10_000_audio_chunks", CONFIG=CONFIG)
    data = torch.utils.data.ConcatDataset([data, no_bird_data])
    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    return train_data, val_data

if __name__ == '__main__':
    torch.manual_seed(CONFIG['seed'])
    data = BirdCLEFDataset(root="/share/acoustic_species_id/BirdCLEF2023_train_audio_chunks")
    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    print(train_data, val_data)