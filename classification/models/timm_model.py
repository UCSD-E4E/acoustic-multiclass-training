# pylint: disable=E1123:

""" Contains the model class
    Model: model with forward pass method. Generated automatically from a timm model

"""
import logging
import config
# timm is a library of premade models
import timm
import torch
from torch import nn

from .model import bce_loss_fn, cross_entropy_loss_fn, focal_loss_fn

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

# pylint: disable=too-many-instance-attributes
class TimmModel(nn.Module):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_classes,
                 model_name="tf_efficientnet_b4",
                 pretrained=True):
        """ Initializes the model
        """
        super().__init__()
        self.num_classes = num_classes
        # See config.py for list of recommended models
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.5)
        self.loss_fn = None
        self.without_logits = cfg.loss_fnc == "BCE"

        logger.debug("add sigmod: %s", str(self.without_logits))
    
    def load_pretrain_checkpoint(self, pretrain_path, remove_pretrained_fc=True):
        #Load in a pretrained model (that used this class)
        pretrained_model = torch.load(pretrain_path)

        #remove potnetially conflicting layers 
        #due to class size differences
        if(remove_pretrained_fc):
            pretrained_model.pop("fc.weight")
            pretrained_model.pop("fc.bias")

        #Load in model so it overwrites only the weights we care about
        self.load_state_dict(pretrained_model, strict=False)
        print("pretrained checkpoint loaded :P")

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
