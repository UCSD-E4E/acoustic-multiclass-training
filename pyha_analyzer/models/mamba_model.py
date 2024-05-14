import pdb

import logging
# timm is a library of premade models
from transformers.models.mamba.modeling_mamba import (
    MambaPreTrainedModel,
    MambaForCausalLM,
    MambaModel,
    ModelOutput,
    MambaCache
)

import torch
import torch.nn as nn

from typing import Optional, Any, Union, Dict, Tuple

from pyha_analyzer import config
from pyha_analyzer.models.base_model import BaseModel

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.sashimi_mamba import Sashimi

cfg = config.cfg
logger = logging.getLogger("acoustic_multiclass_training")

# pylint: disable=too-many-instance-attributes
class MambaAudioModel(BaseModel):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments

    def initialize_model(self, model_name, pretrained, num_classes):
        # TODO: update for class
        # we can improve this
        
        """
        base_model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to('cuda')
        base_model.lm_head = nn.Linear(base_model.config.hidden_size, num_classes, bias=False).to('cuda')
        """
        mamba_config = cfg.model_config['mamba_config']
        model = Sashimi(unet=True,
                        n_layers=mamba_config['n_layers'],
                        pool=[mamba_config['pool']]*2).cuda()
        self.lm_head = nn.Linear(in_features=model.d_output,
                                 out_features=num_classes,
                                 bias=False).to('cuda')
        print('NUM PARAMETERS:', model.count_params())
        return model

    def get_logits_from_model(self, images):
        hs = self.model(images)[0][:, -1]
        return self.lm_head(hs)


class MambaSequenceClassificationOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MambaForSequenceClassification(MambaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = MambaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.score

    def set_output_embeddings(self, new_embeddings):
        self.score = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaSequenceClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        final_state = hidden_states[-1]
        logits = self.score(final_state.to(self.score.weight.dtype)).float()

        loss = None
        if labels is not None:
            raise NotImplementedError

        return MambaSequenceClassificationOutput(
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


    


