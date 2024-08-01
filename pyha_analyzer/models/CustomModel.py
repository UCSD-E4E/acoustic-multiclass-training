import torch
import torch.nn as nn
import json
from torchaudio.models import wav2vec2_model

import logging
# timm is a library of premade models

from pyha_analyzer import config
from pyha_analyzer.models.loss_fn import bce_loss_fn, cross_entropy_loss_fn, focal_loss_fn

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")


class CustomModel(nn.Module):
    def __init__(self, config_path, model_path, num_classes, trainable, embedding_dim=768):
        super(CustomModel, self).__init__()
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.trainable = trainable
        self.freeze_embedding_weights(self.model, trainable)
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        self.loss_fn = None

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def freeze_embedding_weights(self, model, trainable):
        model.feature_extractor.requires_grad_(False)
        model.feature_extractor.eval()
        for param in model.encoder.parameters():
            param.requires_grad = trainable
        if not trainable:
            model.encoder.eval()

    def forward(self, sig):
        out = self.model.extract_features(sig)[0][-1]
        mean_embedding = out.mean(dim=1)
        logits = self.classifier_head(mean_embedding)
        return mean_embedding, logits

    def create_loss_fn(self, cfg, train_dataset):
        loss_desc = cfg.loss_fnc
        if loss_desc == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_desc == "BCE":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise RuntimeError(f"Unsupported loss function: {loss_desc}")

def download_model_files():
    import os
    os.system("wget https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt")
    os.system("wget https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json")
