# pylint: disable=E1123:

""" Contains the model class
    Model: model with forward pass method. Generated automatically from a timm model

"""
import torch
from torch import nn

# timm is a library of premade models
import timm

# pylint: disable=too-many-instance-attributes
class TimmModel(nn.Module):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_classes,
                 model_name="tf_efficientnet_b4",
                 pretrained=True,
                 CONFIG=None):
        """ Initializes the model
        """
        super().__init__()
        self.config = CONFIG
        self.num_classes = num_classes
        # See config.py for list of recommended models
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.loss_fn = None
    
    def forward(self, images):
        """ Forward pass of the model
        """
        return self.model(images)

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        return cross_entropy_loss_fn(self, train_dataset)

def cross_entropy_loss_fn(self,train_dataset):
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

class EarlyStopper:
    """Stop when the model is no longer improving
    """
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience # epochs to wait before stopping
        self.min_delta = min_delta # min change that counts as improvement
        self.counter = 0
        self.max_valid_map = 0

    def early_stop(self, valid_map):
        """ Returns True if the model should early stop
        """
        # reset counter if it improved by more than min_delta
        if valid_map > self.max_valid_map + self.min_delta:
            self.max_valid_map = valid_map
            self.counter = 0
        # increase counter if it has not improved
        elif valid_map < (self.max_valid_map - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
