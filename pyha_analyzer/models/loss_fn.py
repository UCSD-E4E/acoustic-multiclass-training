""" Loss functions """

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss as focal_loss

from pyha_analyzer import config

cfg = config.cfg

def cross_entropy(model, train_dataset, **_):
    """ Returns the cross entropy loss function and sets model.loss_fn
    """
    if not cfg.imb: # normal loss
        model.loss_fn = nn.CrossEntropyLoss()
    else: # weighted loss
        model.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1 / p for p in train_dataset.class_id_to_num_samples.values()]
            ).to(model.device))
    return model.loss_fn

def bce(model, without_logits=False, **_):
    """ Returns the BCE loss function and sets model.loss_fn of model

    Added support for if we want to spilt sigmod and BCE loss or combine with
    BCEwithLogitsLoss
    """
    if not without_logits:
        model.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        model.loss_fn = nn.BCELoss(reduction='mean')
    return model.loss_fn

def laplace(model, **_):
    return (lambda: NotImplementedError)
