import torch
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from examples.llm.src import COMPOSER_MODEL_REGISTRY
import sys
import warnings
from composer.utils import get_device
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

def quantize_model(inference_model, save_model=False, save_path='quantized_model.pt'):
    model = inference_model
    hf_model = model.model
    hf_model = init_compression(hf_model, 'ds_config.json')
    hf_model = redundancy_clean(hf_model, 'ds_config.json')

    model.model = hf_model
    model.eval()
    if save_model:
        print("Saving model to:", save_path)
        torch.save(model, save_path)
    return model


def get_model_from_config(config):
    config.model.init_device = 'cpu'
    model = COMPOSER_MODEL_REGISTRY[config.model.name](config.model,
                                                       config.tokenizer)

    model.eval()
    return model


# Call quantization.py directly to quantize a model from a config and save it.
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('please provide a configuration yaml (first arg) and model save path (second arg).')
        sys.exit(-1)
    yaml_path, save_path = sys.argv[1], sys.argv[2:][0]
    yaml_config = None
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    model = get_model_from_config(yaml_config)
    qm = quantize_model(model, save_model=True, save_path=save_path)
