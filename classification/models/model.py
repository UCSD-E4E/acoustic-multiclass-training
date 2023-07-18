# pylint: disable=E1123:

""" Contains model methods

    note: Models have been moved to thier own python folder
"""
import config
import torch
from torch import nn
from utils import print_verbose

# timm is a library of premade models


cfg = config.cfg

def cross_entropy_loss_fn(self,train_dataset):
    """ Returns the cross entropy loss function and sets self.loss_fn
    """
    print_verbose("CE", cfg.loss_fnc, verbose=cfg.verbose)
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
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        print_verbose("BCEWL", cfg.loss_fnc, verbose=cfg.verbose)
    else:
        self.loss_fn = nn.BCELoss()
        print_verbose("BCE", cfg.loss_fnc, verbose=cfg.verbose)
    return self.loss_fn
