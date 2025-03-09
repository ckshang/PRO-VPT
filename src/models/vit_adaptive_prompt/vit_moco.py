#!/usr/bin/env python3
"""
vit-moco-v3 with prompt: iterative Prompt RelOcation-based VPT
"""
import math
import torch
import torch.nn as nn
import torchvision as tv
import numpy as np

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_moco import VisionTransformerMoCo
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class AdaptivePromptedVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            total_layer = len(self.blocks)

            self.init_prompt_embeddings = nn.Parameter(torch.zeros(total_layer, num_tokens, self.embed_dim))
            nn.init.uniform_(self.init_prompt_embeddings.data, -val, val)  # xavier_uniform initialization
            self.prompts_per_layer = [[] for _ in range(total_layer)]

            self.init_prompt_gates = nn.Parameter(torch.ones(total_layer, num_tokens))
            self.prompt_gates_per_layer = [[] for _ in range(total_layer)]

        else:
            raise ValueError("Other initiation scheme is not supported")

    def allocate_prompts(self):
        for layer_idx in range(len(self.blocks)):
            for prompt_idx in range(self.num_tokens):
                self.prompts_per_layer[layer_idx].append(
                    self.init_prompt_embeddings[layer_idx][prompt_idx].clone().detach().requires_grad_(True)
                )
                self.prompt_gates_per_layer[layer_idx].append(
                    self.init_prompt_gates[layer_idx][prompt_idx].clone().detach().requires_grad_(False)
                )
        del self.init_prompt_embeddings
        del self.init_prompt_gates

    def start_gate_ident(self):
        self.prompt_dropout.requires_grad = False
        if self.prompt_config.PROJECT > -1:
            self.prompt_proj.requires_grad = False
        for layer_idx in range(len(self.prompts_per_layer)):
            for prompt_idx in range(len(self.prompts_per_layer[layer_idx])):
                self.prompts_per_layer[layer_idx][prompt_idx].requires_grad = False
                self.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = True

    def end_gate_ident(self):
        self.prompt_dropout.requires_grad = True
        if self.prompt_config.PROJECT > -1:
            self.prompt_proj.requires_grad = True
        for layer_idx in range(len(self.prompts_per_layer)):
            for prompt_idx in range(len(self.prompts_per_layer[layer_idx])):
                self.prompts_per_layer[layer_idx][prompt_idx].requires_grad = True
                self.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = False

    def prompt_norm_ident(self):
        self.prompt_dropout.requires_grad = True
        if self.prompt_config.PROJECT > -1:
            self.prompt_proj.requires_grad = True
        for layer_idx in range(len(self.prompts_per_layer)):
            for prompt_idx in range(len(self.prompts_per_layer[layer_idx])):
                self.prompts_per_layer[layer_idx][prompt_idx].requires_grad = True
                self.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = False

    def get_prompt_params(self):
        return [prompt for prompts in self.prompts_per_layer for prompt in prompts] + \
               [gate for gates in self.prompt_gates_per_layer for gate in gates]

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((
                cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.embeddings(x)

        num_prev_prompts = 0
        for i in range(len(self.blocks)):
            if len(self.prompts_per_layer[i]) != 0:
                gate_prompt_embeddings = torch.stack([
                    gate * param.view(-1) for gate, param in zip(self.prompt_gates_per_layer[i], self.prompts_per_layer[i])
                ], dim=0)
                gate_prompt_embeddings = gate_prompt_embeddings.unsqueeze(0)
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        gate_prompt_embeddings.expand(x.shape[0], -1, -1)),
                    x[:, 1 + num_prev_prompts:, :]
                ), dim=1)
            else:
                x = torch.cat((
                    x[:, :1, :],
                    x[:, 1 + num_prev_prompts:, :]
                ), dim=1)

            x = self.blocks[i](x)

            num_prev_prompts = len(self.prompts_per_layer[i])

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def get_num_blk(self):
        return len(self.prompts_per_layer)

    def get_prompt_gates_per_layer(self):
        return self.prompt_gates_per_layer

    def get_blk_prompt_norm(self, blk_idx):
        blk_prompt_norm = []
        for prompt_idx in range(len(self.prompts_per_layer[blk_idx])):
            blk_prompt_norm.append(torch.norm(
                self.prompts_per_layer[blk_idx][prompt_idx].detach().cpu()
            ).item())
        return np.array(blk_prompt_norm)

    def get_cur_prompt_distribution(self):
        cur_prompt_distribution = []
        for i in range(len(self.prompts_per_layer)):
            cur_prompt_distribution.append(len(self.prompts_per_layer[i]))
        return cur_prompt_distribution

    def relocation(self, transfer_from_layer, transfer_from_prompt, transfer_to):
        prompt_to_transfer = self.prompts_per_layer[transfer_from_layer].pop(transfer_from_prompt)
        prompt_to_transfer *= 1e-3
        gate_to_transfer = self.prompt_gates_per_layer[transfer_from_layer].pop(transfer_from_prompt)
        self.prompts_per_layer[transfer_to].append(prompt_to_transfer)
        self.prompt_gates_per_layer[transfer_to].append(gate_to_transfer)


def vit_base(prompt_cfg, **kwargs):
    model = AdaptivePromptedVisionTransformerMoCo(
        prompt_cfg,
        patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

