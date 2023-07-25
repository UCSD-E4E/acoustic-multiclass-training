# pylint: disable=E1123:

""" Contains model methods

    note: Models have been moved to their own python folder
"""
import config
import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss as focal_loss

# timm is a library of premade models

cfg = config.cfg

def cross_entropy_loss_fn(self,train_dataset):
    """ Returns the cross entropy loss function and sets self.loss_fn
    """
    if not cfg.imb: # normal loss
        self.loss_fn = nn.CrossEntropyLoss()
    else: # weighted loss
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1 / p for p in train_dataset.class_id_to_num_samples.values()]
            ).to(self.device))
    return self.loss_fn

def bce_loss_fn(self, without_logits=False):
    """ Returns the BCE loss function and sets self.loss_fn of model

    Added support for if we want to spilt sigmod and BCE loss or combine with
    BCEwithLogitsLoss
    """
    if not without_logits:
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        self.loss_fn = nn.BCELoss(reduction='mean')
    return self.loss_fn


def focal_loss_fn(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"):
    """ Loss used in https://arxiv.org/abs/1708.02002. and 1st place winner of birdclef 2023
    Code implementation based heavily on
    https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Focal loss takes BCE loss and uses weight schemes to balance weight between
    - Easier vs harder examples to classify
    - Positive or negative examples
    """
    def focal_loss_temp(
            inputs: torch.Tensor,
            targets: torch.Tensor
    ):
        
        return focal_loss(
            inputs,
            targets,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction
        )
    
    self.loss_fn = focal_loss_temp
    return self.loss_fn
