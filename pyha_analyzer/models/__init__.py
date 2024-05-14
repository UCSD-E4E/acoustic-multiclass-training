import pdb

from .timm_model import TimmModel
from .mamba_model import MambaAudioModel
from pyha_analyzer import config
from pyha_analyzer.models.base_model import BaseModel

cfg = config.cfg

def get_model(num_classes, model_name, pretrained):
    if cfg.use_mamba:
        return MambaAudioModel(num_classes=num_classes,
                               model_name=model_name,
                               pretrained=pretrained)
    else:
        return TimmModel(num_classes=num_classes,
                         model_name=model_name,
                         pretrained=pretrained)

