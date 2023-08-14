""" Contains the model class
    Model: model with forward pass method. Generated automatically from a timm model

"""
import logging
from typing import Callable, Optional, Tuple

# timm is a library of premade models
import timm
import torch
from torch import nn
from torch.amp.autocast_mode import autocast

from pyha_analyzer import config
from pyha_analyzer.models.loss_fn import bce, cross_entropy

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
            drop_rate=cfg.drop_rate)
        self.loss_fn: Optional[Callable] = None
        self.without_logits = cfg.loss_fnc == "BCE"

        logger.debug("add sigmod: %s", str(self.without_logits))

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
        loss_args = {
            "without_logits": self.without_logits,
            "train_dataset" : train_dataset
        }
        loss_functions = {
            "CE"    : cross_entropy(self, **loss_args),
            "BCE"   : bce(self, **loss_args),
            "BCEWL" : bce(self, **loss_args),
        }
        loss_fn = loss_functions[loss_desc]
        return loss_fn

    def try_load_checkpoint(self) -> bool:
        """ Returns true if a checkpoint is specified and loads properly
        Raises an error if checkpoint path is invalid
        Returns true if model checkpoint is loaded """
        if cfg.model_checkpoint == "" or cfg.model_checkpoint is None:
            return False
        try:
            self.load_state_dict(torch.load(cfg.model_checkpoint))
        except FileNotFoundError as exc:
            raise FileNotFoundError("Model not found: " + cfg.model_checkpoint) from exc
        return True

    def run_batch(self, mels: torch.Tensor,
                  labels: torch.Tensor,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Runs the model on a single batch 
            Args:
                model: the model to pass the batch through
                mels: single batch of input data
                labels: single batch of expecte output
            Returns (tuple of):
                loss: the loss of the batch
                outputs: the output of the model
        """
        mels = mels.to(cfg.device)
        labels = labels.to(cfg.device)
        if cfg.device == "cpu":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        with autocast(device_type=cfg.device, dtype=dtype, enabled=cfg.mixed_precision):
            outputs = (self)(mels)
            if self.loss_fn is None:
                raise RuntimeError("Loss function not created")
            # pylint: disable-next=not-callable
            loss = self.loss_fn(outputs, labels)
        outputs = outputs.to(dtype=torch.float32)
        loss = loss.to(dtype=torch.float32)
        assert outputs is not None
        return loss, outputs

    # Temp, only works for efficientnet
    def get_features(self, images):
        """ Get features from an efficientnet model """
        assert images.shape[0]>1, "batch size >1"
        x = self.model.conv_stem(images)
        x = self.model.bn1(x)
        x = self.model.blocks(x)
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        # Was 2
        #[batch_size, *features_dims]
        print(f"{x.shape}")
        x = torch.squeeze(torch.nn.AvgPool2d(x.shape[2:])(x))
        print(f"{x.shape}")

        return x
