""" Loss functions """

import torch
from torch import nn

from pyha_analyzer import config

cfg = config.cfg

def cross_entropy(model, train_dataset, **_):
    """ Returns the cross entropy loss function and sets model.loss_fn
    """
    weight = None
    if cfg.imb and train_dataset is not None:
        weight = get_weights(train_dataset).to(cfg.device)
    model.loss_fn = nn.CrossEntropyLoss(weight=weight)
    return model.loss_fn

def bce(model, train_dataset, without_logits=False, **_):
    """ Returns the BCE loss function and sets model.loss_fn of model

    Added support for if we want to spilt sigmod and BCE loss or combine with
    BCEwithLogitsLoss
    """
    weight = None
    if cfg.imb and train_dataset is not None:
        weight = get_weights(train_dataset).to(cfg.device)

    if not without_logits:
        model.loss_fn = nn.BCEWithLogitsLoss(reduction='none', weight=weight)
    else:
        model.loss_fn = nn.BCELoss(reduction='none', weight=weight)
    return model.loss_fn

def get_weights(dataset):
    return torch.tensor([min(1/p, 1) for p in dataset.class_sums])
