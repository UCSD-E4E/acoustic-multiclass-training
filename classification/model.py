import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchaudio 

import timm

#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter
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
                 model_name="tf_efficientnet_b1", 
                 embedding_size=768, 
                 pretrained=True,
                 CONFIG=None):
        super(BirdCLEFModel, self).__init__()
        self.config = CONFIG
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
    
    def load_pretrain_checkpoint(self, pretrain_path):
        #Load in a pretrained model (that used this class)
        pretrained_model = torch.load(pretrain_path)

        #remove potnetially conflicting layers 
        #due to class size differences
        pretrained_model.pop("fc.weight")
        pretrained_model.pop("fc.bias")

        #Load in model so it overwrites only the weights we care about
        self.load_state_dict(pretrained_model, strict=False)
        print("pretrained checkpoint loaded :P")