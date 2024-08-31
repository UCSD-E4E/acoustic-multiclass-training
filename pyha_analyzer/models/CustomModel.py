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
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(self, cfg, num_classes, model_path, trainable, config_path, embedding_dim=768):
        super().__init__()
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.cfg = cfg
        self.trainable = trainable
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.freeze_embedding_weights(self.model, trainable)
        # Add a linear layer to match the embedding dimensions
        self.embedding_transform = nn.Linear(768, num_classes) #TODO: change this when switching models
        # We will only train the classifier head
        #self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        self.audio_sr = cfg.sample_rate

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def forward(self, sig):
        """
        Input
          sig (Tensor): (batch, time)
        Returns
          mean_embedding (Tensor): (batch, output_dim)
          logits (Tensor): (batch, n_classes)
        """
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        mean_embedding = out.mean(dim=1) #over time
        logits = self.embedding_transform(mean_embedding)  # Transform embedding dimensions
        #logits = self.classifier_head(mean_embedding)
        return mean_embedding, logits

    def freeze_embedding_weights(self, model, trainable):
        """ Freeze weights in AVES embeddings for classification """
        # The convolutional layers should never be trainable
        model.feature_extractor.requires_grad_(False)
        model.feature_extractor.eval()
        # The transformers are optionally trainable
        for param in model.encoder.parameters():
            param.requires_grad = trainable
        if not trainable:
            # We also set layers without params (like dropout) to eval mode, so they do not change
            model.encoder.eval()
    
    def set_eval_aves(model):
        """ Set AVES-based classifier to eval mode. Takes into account whether we are training transformers """
        model.classifier_head.eval()
        model.model.encoder.eval()
        
    

    def create_loss_fn(self, train_dataset):
        loss_desc = self.cfg.loss_fnc
        if loss_desc == "CE":
            return cross_entropy_loss_fn(self, train_dataset)
        if loss_desc == "BCE":
            return bce_loss_fn(self, without_logits=True)
        if loss_desc == "BCEWL":
            return bce_loss_fn(self, without_logits=False)
        if loss_desc == "FL":
            return focal_loss_fn(self, self.without_logits)
        raise RuntimeError("Unsupported loss function")

def download_model_files():
    import os
    os.system("wget https://storage.googleapis.com/esp-public-files/ported_aves/birdaves-biox-base.torchaudio.pt")
    os.system("wget https://storage.googleapis.com/esp-public-files/ported_aves/birdaves-biox-base.torchaudio.model_config.json")
