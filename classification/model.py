import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# timm is a library of premade models
import timm

#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter
class GeM(nn.Module):
    """ Layer that applies 2d Generalized Mean Pooling (GeM) on an input tensor
        Args:
            p: power for generalized mean pooling
            eps: epsilon (avoid zero division)
        
        Layer applies the function ((x_1^p + x_2^p + ... + x_n^p)/n)^(1/p) as compared to max pooling 2d which does something like max(x_1, x_2, ..., x_n)
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        """ Applies generalized mean pooling on an input tensor
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        """ Returns a string representation of the object
        """
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class BirdCLEFModel(nn.Module):
    def __init__(self, 
                 model_name="tf_efficientnet_b4", 
                 embedding_size=768, 
                 pretrained=True,
                 CONFIG=None):
        super().__init__()
        self.config = CONFIG
        # Load in the efficientnet_b4 model preset
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