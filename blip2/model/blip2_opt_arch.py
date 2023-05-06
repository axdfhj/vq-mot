"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import pdb

# from src.models.util import load_pretrained_model

from .blip2 import Blip2Base, disabled_train
from .modeling_opt import OPTForCausalLM



class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(self, opt):
        img_size = opt["img_size"]
        drop_path_rate = opt["drop_path_rate"]
        use_grad_checkpoint = opt["use_grad_checkpoint"]
        vit_precision = opt["vit_precision"]
        freeze_vit = opt["freeze_vit"]
        num_query_token = opt["num_query_token"]
        opt_model = opt["opt_model"]
        prompt = opt["prompt"]
        max_txt_len = opt["max_txt_len"]
        opt_pretrained = opt["opt_model_path"]
        self.temperal_emb = opt.get("temperal_emb", False)
        self.use_nucleus_sampling = opt.get("use_nucleus_sampling", False)
        self.num_captions = opt.get("num_captions", 1)
        
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if self.temperal_emb:
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, 1408)) for _ in range(8)
            )
        if opt.get("freeze_vit", False):
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if opt.get("freeze_qformer", False):
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
        if opt.get("freeze_ln_vision", False):
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
        if opt.get("freeze_query", False):
            self.query_tokens.requires_grad = False

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        if opt.get("freeze_opt_proj", False):
            for name, param in self.opt_proj.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        finetuned_checkpoint = opt.get("finetuned_checkpoint", None)
        if finetuned_checkpoint == None:
            print(f'loading model from {opt_pretrained}!')
            state_dict = torch.load(opt_pretrained, map_location="cpu")
            self.load_state_dict(state_dict['model'], strict=False)
        else:
            print(f'loading model from {finetuned_checkpoint}!')
            state_dict = torch.load(finetuned_checkpoint, map_location="cpu")
            self.load_state_dict(state_dict['model'], strict=False)
        self.num_beams = opt.get("num_beams", 5)

    def forward(self, samples, train):
        if train:
            embeds = []
            for image in samples["image"]:
                image = image.cuda()
                embeds.append(self.ln_vision(self.visual_encoder(image)))
            if self.temperal_emb:
                embeds = [f + e for f, e in zip(embeds, self.img_temperal_embedding)]
            # 1. concat
            image_embeds = torch.cat(embeds, dim=1)
            # 2. mean pooling
            # image_embeds = torch.stack(embeds, dim=1).mean(dim=1)
            # 3. max pooling
            # image_embeds = torch.stack(embeds, dim=1).max(dim=1)[0]

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            self.opt_tokenizer.padding_side = "right"

            text = [self.prompt + " " + t + "\n" for t in samples["text_input"]]

            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if self.prompt:
                targets[
                    :, : self.prompt_length
                ] = -100  # do not apply loss to the prompt

            empty_targets = (
                torch.ones(atts_opt.size(), dtype=torch.long)
                .to(image.device)
                .fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            inputs_embeds = self.opt_model.model.decoder.embed_tokens(
                opt_tokens.input_ids
            )
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return loss
        else:
            return self.generate(samples, num_beams=self.num_beams, use_nucleus_sampling=self.use_nucleus_sampling, num_captions=self.num_captions)

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        embeds = []
        for image in samples["image"]:
            image = image.cuda()
            embeds.append(self.ln_vision(self.visual_encoder(image)))
        if self.temperal_emb:
            embeds = [f + e for f, e in zip(embeds, self.img_temperal_embedding)]
        # 1. concat
        image_embeds = torch.cat(embeds, dim=1)
        # 2. mean pooling
        # image_embeds = torch.stack(embeds, dim=1).mean(dim=1)
        # # 3. max pooling
        # image_embeds = torch.stack(embeds, dim=1).max(dim=1)[0]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * image.size(0)

        opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        if use_nucleus_sampling:
            query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            num_beams = 1
        else:
            query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)
        outputs = self.opt_model.generate(
            input_ids=input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        prompt_length = opt_tokens.input_ids.shape[1]
        output_text = self.opt_tokenizer.batch_decode(
            outputs[:, prompt_length:], skip_special_tokens=True
        )
        output_text = [text.strip() for text in output_text]
        return output_text
