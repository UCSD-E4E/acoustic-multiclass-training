# pylint: disable=E1123:

""" Contains the model class
    Model: model with forward pass method. Generated automatically from a timm model

"""
import torch
from torch import nn
from utils import print_verbose
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
        self.without_logits = self.config.loss_fnc == "BCE"

        print_verbose("add sigmod: ", self.without_logits, verbose=self.config.verbose)
    
    def forward(self, images):
        """ Forward pass of the model
        """
        x = self.model(images)
        if self.without_logits:
            x = torch.sigmoid(x)
        return x

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        loss_desc = self.config.loss_fnc
        if loss_desc == "CE":
            return cross_entropy_loss_fn(self, train_dataset)#
        if loss_desc == "BCE":
            return bce_loss_fn(self, self.without_logits)
        if loss_desc == "BCEWL":
            return bce_loss_fn(self, self.without_logits)
        raise RuntimeError("Pick a loss in the form of CE, BCE, BCEWL")

def cross_entropy_loss_fn(self,train_dataset):
    """ Returns the loss function and sets self.loss_fn
    """
    print_verbose("CE", self.config.loss_fnc, verbose=self.config.verbose)
    if not self.config.imb: # normal loss
        self.loss_fn = nn.CrossEntropyLoss()
    else: # weighted loss
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1 / p for p in train_dataset.class_id_to_num_samples.values()]
            ).to(self.device))
    return self.loss_fn

def bce_loss_fn(self, without_logits=False):
    if not without_logits:
        self.loss_fn = nn.BCEWithLogitsLoss()
        print_verbose("BCEWL", self.config.loss_fnc, verbose=self.config.verbose)
    else:
        self.loss_fn = nn.BCELoss()
        print_verbose("BCE", self.config.loss_fnc, verbose=self.config.verbose)
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
