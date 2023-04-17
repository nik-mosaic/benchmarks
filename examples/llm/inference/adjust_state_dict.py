# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import sys
import warnings

import torch
from composer.core import get_precision_context
from composer.utils import get_device
from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY
from examples.llm.inference.quantization import quantize_model

import time

def update_state_dict(checkpoint_yaml_path: str, save_to_file: bool = False):
    print("Updating state dict.")
    with open(checkpoint_yaml_path) as f:
        cfg = om.load(f)
    # set init_device to cpu for checkpoint loading
    cfg.model.init_device = 'cpu'
    model = build_composer_model(cfg.model, cfg.tokenizer)

    ckpt_load_path = cfg.get('load_path', None)  # type: ignore
    if ckpt_load_path is None:
        raise ValueError('Checkpoint load_path is required for importing.')

    checkpoint = torch.load(ckpt_load_path, map_location='cpu')

    state_dict = checkpoint['state']['model']

    adjusted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if 'causal_attn.W_qkv' in key:
            new_key = key.replace('causal_attn.W_qkv', 'attn.Wqkv')
        if 'causal_attn.out_proj' in key:
            new_key = key.replace('causal_attn.out_proj', 'attn.out_proj')
        if 'causal_attn.k_ln' in key:
            new_key = key.replace('causal_attn.k_ln', 'attn.k_ln')
        if 'causal_attn.q_ln' in key:
            new_key = key.replace('causal_attn.q_ln', 'attn.q_ln')
        adjusted_state_dict[new_key] = value

    checkpoint['state']['model'] = adjusted_state_dict

    if save_to_file:
        print("Saving new model.")
        torch.save(checkpoint, "/root/13b-checkpoint.pt.adjusted")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please provide a configuration yaml.')
        sys.exit(-1)
    yaml_path = sys.argv[1]

    update_state_dict(yaml_path, save_to_file=True)
    