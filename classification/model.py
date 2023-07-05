# pylint: disable=E1123:
# Litteraly I don't know why this code contains a pos_weight for CEL
# This code comes from the BIRDCLEF2023 code, a review is despertely needed to understand why
# This is being done to the loss function. 

""" Contains the model class and the Generalized Mean Pooling layer

    GeM: generalized mean pooling layer
    BirdCLEFModel: model with forward pass method

"""
import torch
from torch import nn
import torch.nn.functional as F

# timm is a library of premade models
import timm

#https://www.kaggle.com/code/debarshichanda/pytorch-w-b-birdclef-22-starter
class GeM(nn.Module):
    """ Layer that applies 2d Generalized Mean Pooling (GeM) on an input tensor
        Args:
            p: power for generalized mean pooling
            eps: epsilon (avoid zero division)
        
        Layer applies the function ((x_1^p + x_2^p + ... + x_n^p)/n)^(1/p)
        as compared to max pooling 2d which does something like max(x_1, x_2, ..., x_n)
    """
    def __init__(self, p=3, eps=1e-6):
        """ Initializes the layer
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        """ Forward pass of the layer
        """
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        """ Applies generalized mean pooling on an input tensor
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        """ Returns a string representation of the object
        """
        return self.__class__.__name__ + \
                '(' + f'p={self.p.data.tolist()[0]:.4f}'+ \
                ', ' + 'eps=' + str(self.eps) + ')'

class BirdCLEFModel(nn.Module):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments
    def __init__(self, 
                 num_classes,
                 model_name="tf_efficientnet_b4", 
                 embedding_size=768, 
                 pretrained=True,
                 CONFIG=None):
        """ Initializes the model
        """
        super().__init__()
        self.config = CONFIG
        self.num_classes = num_classes
        # Load in the efficientnet_b4 model preset
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.loss_fn = None
    
    def forward(self, images):
        """ Forward pass of the model
        """
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding)
        return output
    
    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        if not self.config.imb: # normal loss
            self.loss_fn = nn.CrossEntropyLoss()
        else: # weighted loss
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [1 / p for p in train_dataset.class_id_to_num_samples.values()]
                ).to(self.device))
        return self.loss_fn
