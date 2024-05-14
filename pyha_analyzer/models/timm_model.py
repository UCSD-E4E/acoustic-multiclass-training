import logging
# timm is a library of premade models
import timm

from pyha_analyzer import config
from pyha_analyzer.models.base_model import BaseModel

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

# pylint: disable=too-many-instance-attributes
class TimmModel(BaseModel):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments

    def initialize_model(self, model_name, pretrained, num_classes):
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=cfg.drop_rate)

    def get_logits_from_model(self, images):
        return self.model(images)
