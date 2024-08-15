from typing import Optional
from jaxtyping import Float, Int

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import torch as t
import numpy as np

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
        'unk_token': 'UNK',
        'bos_token': 'BOS',
        'pad_token': 'PAD',
    })
    return hf_tokenizer


