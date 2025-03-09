#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
optimizer, ref:
https://github.com/huggingface/transformers/blob/master/transformers/optimization.property  #noqa
"""
import math

import torch
from fvcore.common.config import CfgNode
from torch.optim import Optimizer
import torch.optim as optim
from typing import Any, Callable, Iterable, List, Tuple, Optional

from ..utils import logging
logger = logging.get_logger("visual_prompt")


def make_optimizer(
    models: List[Any], train_params: CfgNode, additional_params=None
) -> Optimizer:
    params = []
    for model in models:
        # only include learnable params
        if train_params.DBG_TRAINABLE:
            logger.info("Trainable params:")

        for key, value in model.named_parameters():

            if value.requires_grad:

                if train_params.DBG_TRAINABLE:
                    logger.info("\t{}, {}, {}".format(key, value.numel(), value.shape))
                params.append((key, value))

    if train_params.WEIGHT_DECAY > 0:
        if train_params.OPTIMIZER == 'adamw':

            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            no_decay = ['bias']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': train_params.WEIGHT_DECAY},
                {'params': [p for n, p in params
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if additional_params is not None:
                optimizer_grouped_parameters += [
                    {'params': additional_params,
                     'weight_decay': train_params.WEIGHT_DECAY}
                ]
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=train_params.BASE_LR,
                weight_decay=train_params.WEIGHT_DECAY
            )
        else:
            _params = []
            for p in params:
                key, value = p
                # print(key)
                # if not value.requires_grad:
                #     continue
                lr = train_params.BASE_LR
                weight_decay = train_params.WEIGHT_DECAY
                # if "last_layer.bias" in key:  # for employing mlp
                #     # no regularization (weight decay) for last layer's bias
                #     weight_decay = 0.0

                # if train_params.BIAS_MULTIPLIER == 1.:
                _params += [{
                    "params": [value],
                    "lr": lr,
                    "weight_decay": weight_decay
                }]
                # else:  # for employing bias
                #     if "bias" in key and "last_layer.bias" not in key:
                #         # use updated lr for this param
                #         lr_value = lr * train_params.BIAS_MULTIPLIER
                #     else:
                #         lr_value = lr
                #
                #     if train_params.DBG_TRAINABLE:
                #         logger.info("\t{}, {:.4f}".format(key, lr_value))
                #
                #     _params += [{
                #         "params": [value],
                #         "lr": lr_value,
                #         "weight_decay": weight_decay
                #     }]

            if additional_params is not None:
                _params += [
                    {"params": additional_params,
                     "lr": train_params.BASE_LR,
                     "weight_decay": train_params.WEIGHT_DECAY}
                ]

            # if train_params.OPTIMIZER == 'adam':
            #     optimizer = optim.Adam(
            #         _params,
            #         lr=train_params.BASE_LR,
            #         weight_decay=train_params.WEIGHT_DECAY,
            #     )
            # else:
            optimizer = optim.SGD(
                _params,
                train_params.BASE_LR,
                momentum=train_params.MOMENTUM,
                weight_decay=train_params.WEIGHT_DECAY
            )
        return optimizer
    else:
        # if train_params.OPTIMIZER == 'adam':
        #     optimizer = optim.Adam(
        #         model.parameters(),
        #         lr=train_params.BASE_LR
        #     )
        # else:
        _params = []
        for p in params:
            key, value = p

            lr = train_params.BASE_LR

            # if train_params.BIAS_MULTIPLIER == 1.:
            _params += [{
                "params": [value],
                "lr": lr,
            }]
            # else:  # for employing bias
            #     if "bias" in key and "last_layer.bias" not in key:
            #         # use updated lr for this param
            #         lr_value = lr * train_params.BIAS_MULTIPLIER
            #     else:
            #         lr_value = lr
            #
            #     if train_params.DBG_TRAINABLE:
            #         logger.info("\t{}, {:.4f}".format(key, lr_value))
            #
            #     _params += [{
            #         "params": [value],
            #         "lr": lr_value,
            #     }]
        if additional_params is not None:
            _params += [
                {"params": additional_params,
                 "lr": train_params.BASE_LR}
            ]
        optimizer = optim.SGD(
            _params,
            train_params.BASE_LR,
            momentum=train_params.MOMENTUM,
        )
        return optimizer
