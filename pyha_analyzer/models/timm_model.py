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
        self.model_name = model_name
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

    def get_features(self, images):
        """ Get features for a batch of images """
        feature_fn_map = {
            "tf_efficientnet_b4": self.__tf_efficientnet_b4_features,
            "eca_nfnet_l0"      : self.__eca_nfnet_l0_features,
        }
        if self.model_name not in feature_fn_map:
            raise NotImplementedError(
                f"Feature function not implemented for {self.model_name}"
            )
        feature_fn = feature_fn_map[self.model_name]
        return feature_fn(images)

    def __tf_efficientnet_b4_features(self, images):
        """ Get features from tf_efficientnet_b4 model """
        assert images.shape[0]>1, "batch size >1"
        model = self.model
        x = torch.nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.blocks,
                model.conv_head,
                model.bn2,
        )(images) #[batch_size, *features_dims]
        if cfg.features_flattened:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        else:
            x = torch.squeeze(
                torch.nn.AvgPool2d(x.shape[2:])(x)
            ) # Squeeze feature dims to 1D
        return x

    def __eca_nfnet_l0_features(self, images):
        """ Get features from eca_nfnet_l0 model"""
        model = self.model
        x = torch.nn.Sequential(
            model.stem,
            model.stages,
            model.final_conv,
            model.final_act,
        )(images)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        if cfg.features_flattened:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        else:
            x = torch.squeeze(
                torch.nn.AvgPool2d(x.shape[2:])(x)
            ) # Squeeze feature dims to 1D
        return x
