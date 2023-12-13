#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
from abc import ABC

import torch
import yaml
from transformers.modeling_outputs import BaseModelOutput
from models.htsat import HTSAT_Swin_Transformer
from transformers import PreTrainedModel
from models.audio_encoder_config import AudioEncoderConfig
from IPython import embed

class AudioEncoderModel(PreTrainedModel):
    config_class = AudioEncoderConfig

    def __init__(self, config):
        super(AudioEncoderModel, self).__init__(config)

        self.audio_enc = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            patch_stride=(4, 4),
            num_classes=527,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            config=config
        )
        if config.pretrained:
            audio_ckpt = torch.load("/scratch/project_2003370/james/wavcaps/pretrained_models/audio_encoders/HTSAT.ckpt", map_location="cpu")["state_dict"]

            for key in list(audio_ckpt.keys()):
                if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                    and 'logmel_extractor' not in key):
                    v = audio_ckpt.pop(key)
                    audio_ckpt[key[10:]] = v
            self.audio_enc.load_state_dict(audio_ckpt, strict=False)

        self.audio_width = 768


        if config.freeze:
            for name, param in self.audio_enc.named_parameters():
                if "fc1" not in name:
                    param.requires_grad = False

    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):
        # print('\ncheck when enters audio_encoder') 
        # embed()
        audio_embeds = self.audio_enc(input_ids)
        if not return_dict:
            return audio_embeds
        return BaseModelOutput(audio_embeds, None, None)


if __name__ == '__main__':
    import os
    os.chdir("../")
    with open("settings/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = AudioEncoderConfig(**config["audio_encoder_args"], audio_args=config["audio_args"])
    model = AudioEncoderModel(config)
    print(model)

