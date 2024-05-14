import torch.nn as nn

""" Contains the model class
    Model: model with forward pass method. Generated automatically from a timm model

"""
import logging
# timm is a library of premade models
import torch
from torch import nn

from pyha_analyzer import config
from pyha_analyzer.models.loss_fn import bce_loss_fn, cross_entropy_loss_fn, focal_loss_fn

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

class BaseModel(nn.Module):
    """
    Efficient net neural network
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_classes,
                 model_name="tf_efficientnet_b4",
                 pretrained=True):
        """
        Initializes the model
        """
        super().__init__()
        self.num_classes = num_classes
        # See config.py for list of recommended models
        self.model = self.initialize_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.loss_fn = None
        self.without_logits = cfg.loss_fnc == "BCE"

        logger.debug("add sigmoid: %s", str(self.without_logits))

    def initialize_model(self, model_name, pretrained, num_classes):
        raise NotImplementedError

    def get_logits_from_model(self, images):
        raise NotImplementedError
    
    def forward(self, images):
        """ Forward pass of the model
        """
        x = self.get_logits_from_model(images)
        if self.without_logits:
            x = torch.sigmoid(x)
        return x

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        loss_desc = cfg.loss_fnc
        if loss_desc == "CE":
            return cross_entropy_loss_fn(self, train_dataset)#
        if loss_desc == "BCE":
            return bce_loss_fn(self, self.without_logits)
        if loss_desc == "BCEWL":
            return bce_loss_fn(self, self.without_logits)
        if loss_desc == "FL":
            return focal_loss_fn(self, self.without_logits)
        raise RuntimeError("Pick a loss in the form of CE, BCE, BCEWL, or FL")