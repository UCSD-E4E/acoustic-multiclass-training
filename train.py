import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchaudio 

import timm

import argparse
import librosa
import os
import numpy as np
from dataset import BirdCLEFDataset, get_datasets
import wandb
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=264, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=16, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=16, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)


#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training
# generalize mean pooling
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class BirdCLEFModel(nn.Module):
    def __init__(self, 
                 model_name="tf_efficientnet_b4_ns", 
                 embedding_size=768, 
                 pretrained=True):
        super(BirdCLEFModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = nn.Linear(embedding_size, CONFIG.num_classes)
    
    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding)
        return output


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
    
def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()

    running_loss = 0
    log_n = 0
    log_loss = 0
    correct = 0
    total = 0
    # loop = tqdm(data_loader, position=0)
    for i, (mels, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        log_loss += loss.item()
        log_n += 1

        if i % (20) == 0 or i == len(data_loader) - 1:
            print("Loss:", log_loss, "Accuracy:", correct / total * 100.)
            log_loss = 0
            log_n = 0
            correct = 0
            total = 0
        
        # loop.set_description(f"Epoch [{epoch+1}/{CONFIG.epochs}]")
        # loop.set_postfix(loss=loss.item())

    return running_loss/len(data_loader)
    


def set_seed():
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

if __name__ == '__main__':
    CONFIG = parser.parse_args()
    set_seed()
    print("Loading Model...")
    model = BirdCLEFModel().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    print("Model / Optimizer Loading Succesful :P")

    print("Loading Dataset")
    train_dataset, val_dataset = get_datasets(CONFIG=CONFIG)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs
    )
    

    print("Training")
    for epoch in range(CONFIG.epochs):
        train_loss = train(
            model, 
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch)

    print(":o wow")