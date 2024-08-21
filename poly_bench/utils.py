import pathlib
from dataclasses import asdict
from typing import Optional
from jaxtyping import Float, Int

import yaml
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import torch as t
import numpy as np
from huggingface_hub import upload_folder, hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig
from safetensors.torch import save_file, load_file

class CustomDataset(Dataset):
    def __init__(self, 
                 inputs: Int[np.ndarray, "batch seq"], 
                 targets: Float[np.ndarray, "batch seq d_vocab"],
                 markers: Optional[Int[np.ndarray, "batch seq"]] = None
                 ):
        self.inputs = t.tensor(inputs).to(int)
        self.targets = t.tensor(targets).to(t.float32)
        if markers is None:
            self.markers = None
        else:
            self.markers = t.tensor(markers).to(int)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.markers is None:
            return self.inputs[idx], self.targets[idx]
        else:
            return self.inputs[idx], self.targets[idx], self.markers[idx]

def create_tokenizer(vocab: dict) -> PreTrainedTokenizerFast:
    # Create a Tokenizer with a WordLevel model
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="UNK"))
    
    # Set the normalizer, pre-tokenizer, and decoder
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Convert to Hugging Face tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    # Add the special tokens to the Hugging Face tokenizer
    hf_tokenizer.add_special_tokens({
        'bos_token': 'BOS',
        'pad_token': 'PAD',
    })
    return hf_tokenizer

def save_model_to_dir(model: HookedTransformer, local_dir: str):
    directory = pathlib.Path(local_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), directory / 'model.safetensors')
    config_dict = {}
    keys = ['d_model', 'n_layers', 'd_vocab', 'n_ctx', 'd_head', 'seed', 'act_fn']
    for k in keys:
        config_dict[k] = getattr(model.cfg, k)
    with open(directory / 'config.yaml', 'w') as f:
        yaml.dump(config_dict, f)


def save_to_hf(local_dir: str, message: str, repo_name: str = 'evanhanders/polysemantic-benchmarks'):
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_name,
        commit_message=message
    )

def load_from_hf(
        model_name: str, 
        model_file: str = 'model.safetensors', 
        config_file : str = 'config.yaml', 
        repo_name: str = 'evanhanders/polysemantic-benchmarks'
        ) -> None:
    model_tensors_file = hf_hub_download(repo_id=repo_name, filename=f'{model_name}/{model_file}')
    model_config_file = hf_hub_download(repo_id=repo_name, filename=f'{model_name}/{config_file}')
    with open(model_config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = HookedTransformerConfig(**config_dict)
    model = HookedTransformer(cfg)
    model.load_state_dict(load_file(model_tensors_file))
    return model

    
