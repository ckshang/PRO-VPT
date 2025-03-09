#!/usr/bin/env python3
"""
vit with prompt: iterative Prompt RelOcation-based VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import numpy as np

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class AdaptivePromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        super(AdaptivePromptedTransformer, self).__init__(
            config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens per layer

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # # if project the prompt embeddings
        # if self.prompt_config.PROJECT > -1:
        #     # only for prepend / add
        #     prompt_dim = self.prompt_config.PROJECT
        #     self.prompt_proj = nn.Linear(
        #         prompt_dim, config.hidden_size)
        #     nn.init.kaiming_normal_(
        #         self.prompt_proj.weight, a=0, mode='fan_out')
        # else:
        prompt_dim = config.hidden_size
        self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            total_layer = config.transformer["num_layers"]

            self.init_prompt_embeddings = nn.Parameter(torch.zeros(total_layer, num_tokens, prompt_dim))
            nn.init.uniform_(self.init_prompt_embeddings.data, -val, val)  # xavier_uniform initialization
            self.prompts_per_layer = [[] for _ in range(total_layer)]

            self.init_prompt_gates = nn.Parameter(torch.ones(total_layer, num_tokens))
            self.prompt_gates_per_layer = [[] for _ in range(total_layer)]

        else:
            raise ValueError("Other initiation scheme is not supported")

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(self, x):
        attn_weights = []

        x = self.embeddings(x)

        num_prev_prompts = 0
        for i in range(self.vit_config.transformer["num_layers"]):
            if len(self.prompts_per_layer[i]) != 0:
                gate_prompt_embeddings = torch.stack([
                    gate * param.view(-1) for gate, param in zip(self.prompt_gates_per_layer[i], self.prompts_per_layer[i])
                ], dim=0)
                gate_prompt_embeddings = gate_prompt_embeddings.unsqueeze(0)
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_proj(gate_prompt_embeddings).expand(x.shape[0], -1, -1)),
                    x[:, 1 + num_prev_prompts:, :]
                ), dim=1)
            else:
                x = torch.cat((
                    x[:, :1, :],
                    x[:, 1 + num_prev_prompts:, :]
                ), dim=1)

            x, weights = self.encoder.layer[i](x)

            if self.encoder.vis:
                attn_weights.append(weights)

            num_prev_prompts = len(self.prompts_per_layer[i])

        encoded = self.encoder.encoder_norm(x)
        return encoded, attn_weights


class AdaptivePromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        super(AdaptivePromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using AdaptivePromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = AdaptivePromptedTransformer(prompt_cfg, vit_cfg, img_size, vis)

    def allocate_prompts(self):
        for layer_idx in range(self.transformer.vit_config.transformer["num_layers"]):
            for prompt_idx in range(self.transformer.num_tokens):
                self.transformer.prompts_per_layer[layer_idx].append(
                    self.transformer.init_prompt_embeddings[layer_idx][prompt_idx].clone().detach().requires_grad_(True)
                )
                self.transformer.prompt_gates_per_layer[layer_idx].append(
                    self.transformer.init_prompt_gates[layer_idx][prompt_idx].clone().detach().requires_grad_(False)
                )
        del self.transformer.init_prompt_embeddings
        del self.transformer.init_prompt_gates

    def start_gate_ident(self):
        self.head.requires_grad = False
        self.transformer.prompt_dropout.requires_grad = False
        if self.transformer.prompt_config.PROJECT > -1:
            self.transformer.prompt_proj.requires_grad = False
        for layer_idx in range(len(self.transformer.prompts_per_layer)):
            for prompt_idx in range(len(self.transformer.prompts_per_layer[layer_idx])):
                self.transformer.prompts_per_layer[layer_idx][prompt_idx].requires_grad = False
                self.transformer.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = True

    def end_gate_ident(self):
        self.head.requires_grad = True
        self.transformer.prompt_dropout.requires_grad = True
        if self.transformer.prompt_config.PROJECT > -1:
            self.transformer.prompt_proj.requires_grad = True
        for layer_idx in range(len(self.transformer.prompts_per_layer)):
            for prompt_idx in range(len(self.transformer.prompts_per_layer[layer_idx])):
                self.transformer.prompts_per_layer[layer_idx][prompt_idx].requires_grad = True
                self.transformer.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = False

    def start_pruning(self):
        self.head.requires_grad = True
        self.transformer.prompt_dropout.requires_grad = True
        if self.transformer.prompt_config.PROJECT > -1:
            self.transformer.prompt_proj.requires_grad = True
        for layer_idx in range(len(self.transformer.prompts_per_layer)):
            for prompt_idx in range(len(self.transformer.prompts_per_layer[layer_idx])):
                self.transformer.prompts_per_layer[layer_idx][prompt_idx].requires_grad = True
                self.transformer.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = True

    def prompt_norm_ident(self):
        self.head.requires_grad = True
        self.transformer.prompt_dropout.requires_grad = True
        if self.transformer.prompt_config.PROJECT > -1:
            self.transformer.prompt_proj.requires_grad = True
        for layer_idx in range(len(self.transformer.prompts_per_layer)):
            for prompt_idx in range(len(self.transformer.prompts_per_layer[layer_idx])):
                self.transformer.prompts_per_layer[layer_idx][prompt_idx].requires_grad = True
                self.transformer.prompt_gates_per_layer[layer_idx][prompt_idx].requires_grad = False

    def get_prompt_params(self):
        return [prompt for prompts in self.transformer.prompts_per_layer for prompt in prompts] + \
               [gate for gates in self.transformer.prompt_gates_per_layer for gate in gates]

    def get_num_blk(self):
        return len(self.transformer.prompts_per_layer)

    def get_prompt_gates_per_layer(self):
        return self.transformer.prompt_gates_per_layer

    def get_blk_prompt_norm(self, blk_idx):
        blk_prompt_norm = []
        for prompt_idx in range(len(self.transformer.prompts_per_layer[blk_idx])):
            blk_prompt_norm.append(torch.norm(
                self.transformer.prompts_per_layer[blk_idx][prompt_idx].detach().cpu()
            ).item())
        return np.array(blk_prompt_norm)

    def get_cur_prompt_distribution(self):
        cur_prompt_distribution = []
        for i in range(len(self.transformer.prompts_per_layer)):
            cur_prompt_distribution.append(len(self.transformer.prompts_per_layer[i]))
        return cur_prompt_distribution

    def relocation(self, transfer_from_layer, transfer_from_prompt, transfer_to):
        if transfer_from_layer != transfer_to:
            prompt_to_transfer = self.transformer.prompts_per_layer[transfer_from_layer].pop(transfer_from_prompt)
            prompt_to_transfer *= 1e-3
            gate_to_transfer = self.transformer.prompt_gates_per_layer[transfer_from_layer].pop(transfer_from_prompt)

            self.transformer.prompts_per_layer[transfer_to].append(prompt_to_transfer)
            self.transformer.prompt_gates_per_layer[transfer_to].append(gate_to_transfer)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
