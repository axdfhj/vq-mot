"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer

from .eva_vit import create_eva_vit_g
from .Qformer import BertConfig, BertLMHeadModel


class Blip2Base(nn.Module):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width):
        encoder_config = BertConfig.from_pretrained(
            "bert-base-uncased"
        )
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        ln_vision = nn.LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
