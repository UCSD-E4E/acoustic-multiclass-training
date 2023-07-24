""" Loss functions """

import torch
from torch import nn
from pyha_analyzer import config

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
        self.loss_fn = nn.BCEWithLogitsLoss()
    else:
        self.loss_fn = nn.BCELoss()
    return self.loss_fn
