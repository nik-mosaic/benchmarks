# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import copy
import warnings

import deepspeed
import numpy as np
import torch
# You can use this to load the model weights
from composer.core import get_precision_context
from composer.utils import get_device
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from examples.llm.src import COMPOSER_MODEL_REGISTRY
from examples.llm.inference.quantization import quantize_model
from torch.profiler import ProfilerActivity, profile

from examples.llm.src.models.mosaic_gpt import MosaicGPT
from examples.llm.src.models.layers import GPTBlock
import os

def main(config, tp=False):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    input_lengths = [32]
    output_lengths = [128]
    num_runs = 5
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    inf_config = {
        'replace_with_kernel_inject': True,
        'dtype': torch.float16,
        'enable_cuda_graph': True,
        'replace_method': 'auto',
        'tensor_parallel': {
            'tp_size': 0
        },
    }
    if tp:
        assert world_size > 1, "Trying to run Tensor Parallelism with World Size=1"
        print("***** Tensor Parallelism with World Size:", world_size)
        # Must disable replace_with_kernel_inject for tensor parallelism
        inf_config = {
            'dtype': torch.float16,
            'tensor_parallel': {
                'tp_size': world_size
            },            
            'injection_policy': {GPTBlock: ('attn.out_proj', 'mlp.mlp_down')}
        }

    if config.use_quantized_model:      
        quantized_config = copy.deepcopy(inf_config)
        quantized_config['dtype'] = torch.int8
        inf_config = quantized_config
    else:
        print(inf_config)
        print("***** Using REGULAR model from", config.load_path)
 

    config.model.init_device = 'cpu'

    model = COMPOSER_MODEL_REGISTRY[config.model.name](config.model,
                                                       config.tokenizer)
    
    model.optimizers = None
    if model.tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.tokenizer.padding_side = 'left'

    model.eval()
    
    print("HF model:", model.model)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    if config.use_quantized_model:
        print("***** Quantizing model *****.")
        # DeepSpeed quantization
        # model = model.bfloat16()
        model = quantize_model(model)

    # Deepspeed's init_inference takes in a huggingface model, which is the .model
    # object of our ComposerModel object.
    ds_engine = deepspeed.init_inference(model.model, config=inf_config)
    model.model = ds_engine.module

    print("Deepspeed model for inference:", model)

    # Checking if deepspeed casts dtypes correctly
    print("----- DTYPE CHECK -----")
    for n, p in model.named_parameters():
        print('n is: ', n)
        print('dtype is: ', p.dtype)
        break

    stats = []
    print('Run Name\tLatency\tTokens per Second')
    for batch_size in batch_sizes:
        for input_length in input_lengths:
            for output_length in output_lengths:
                times = []
                eos_token = tokenizer.eos_token
                # Make sure we are not generating a fake batch with a EOS token
                while True:
                    batch = torch.randint(
                        0,
                        config.model.vocab_size - 1,
                        size=(batch_size, input_length
                             )).to(f'cuda:{torch.cuda.current_device()}')
                    if tokenizer.convert_tokens_to_ids(eos_token) not in batch:
                        break
                batch = batch.to(torch.long)
                torch.cuda.synchronize()
                for i in range(num_runs + 1):
                    start_time = time.time()
                    with torch.no_grad():
                        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
                        model.model.generate(
                            batch,
                            max_new_tokens=output_length,
                            use_cache=True)
                    torch.cuda.synchronize()
                    # We noticed there sometimes might be a small bit of startup time
                    # so we only start to benchmark after the first iteration
                    if i > 0:
                        times.append(time.time() - start_time)

                #print(prof.key_averages(group_by_stack_n=10).table(sort_by="cpu_time_total", row_limit=250))
                #prof.export_chrome_trace('trace-deepspeed-256.json')
                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)

                resu = (
                    f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}',
                    f'{mean_time:.3f}', f'{tokens_per_second:.3f}')

                run_name, latency, tokens_per_second = resu

                print(f'{run_name}\t\t{latency}\t\t{tokens_per_second}')

                stats.append(resu)

    print('=' * 75)
    print('name, latency (s), tokens / s')
    for val in stats:
        print(val)


if __name__ == '__main__':
    enable_tensor_parallelism = True

    yaml_path, args_list = sys.argv[1], sys.argv[2:]    
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    cli_config = om.from_cli(args_list)
    config = om.merge(yaml_config, cli_config)
    print(config)
    main(config, tp=enable_tensor_parallelism)
