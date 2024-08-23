from abc import ABC, abstractmethod
from typing import Optional

import torch as t
import numpy as np
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookedRootModule # type: ignore
from transformer_lens import HookedTransformerConfig, HookedTransformer # type: ignore
from transformer_lens.utils import get_device # type: ignore
from datasets import Dataset # type: ignore
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers # type: ignore
from transformers import PreTrainedTokenizerFast # type: ignore

from iit.utils.iit_dataset import train_test_split, IITDataset # type: ignore
from iit.utils.correspondence import Correspondence # type: ignore

from ..utils import SimpleDataset


class PolyCase(HookedRootModule, ABC):
    def __init__(self, vocab_dict: Optional[dict[str, int]] = None, device: str = get_device()):
        if vocab_dict is None:
            raise ValueError("vocab_dict must be provided")
        super().__init__()
        self.vocab_dict = vocab_dict
        self.d_vocab = len(vocab_dict.keys())
        self.device = device

    @abstractmethod
    def get_ll_model_cfg(self) -> HookedTransformerConfig:
        pass
    
    def get_ll_model(self, cfg: Optional[HookedTransformerConfig] = None, seed: Optional[int] = None) -> HookedTransformer:
        if cfg is None:
            cfg = self.get_ll_model_cfg()
        if seed is not None:
            cfg.seed = seed
        cfg.init_mode = "xavier_normal"
        return HookedTransformer(cfg)

    @abstractmethod
    def get_correspondence(self) -> Correspondence:
        pass
         
    def is_categorical(self) -> bool:
        return True

    @abstractmethod
    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Tensor, t.Tensor]) -> Float[t.Tensor, "batch seq logits"]:
        pass


class PolyBenchDataset(ABC):
    
    def __init__(
        self, 
        N_samples: int, 
        map_dict: Optional[dict[str, int]] = None,
        n_ctx: int = 15,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.N_samples = N_samples
        self.n_ctx = n_ctx
        if map_dict is None:
            raise ValueError("map_dict must be provided")
        self.map_dict = map_dict
        self.reverse_map_dict: dict[int, str] = {v: k for k, v in map_dict.items()}

        self.tokens: np.ndarray | t.Tensor = np.array([])
        self.labels: np.ndarray | t.Tensor = np.array([])
        self.markers: np.ndarray | t.Tensor = np.array([])
        self.str_tokens = np.array([])

        #generate lists of self.str_tokens and self.tokens:
        self.generate_tokens()
        self.map_tokens_to_str()
        self.generate_labels()
        self.build_dataset()
    
    def build_dataset(self) -> None:
        self.dataset = Dataset.from_dict({
                'tokens' : self.tokens,
                'str_tokens' : self.str_tokens,
                'labels': self.labels,
                'markers' : self.markers
            })
  
    @abstractmethod
    def generate_tokens(self) -> None:
        pass

    def map_tokens_to_str(self) -> None:
        # Vectorized mapping using numpy
        vectorized_map = np.vectorize(self.map_dict.get)
        self.str_tokens = vectorized_map(self.tokens)

    @abstractmethod
    def generate_labels(self, skip_first: bool = False) -> None:
        pass

    def get_dataset(self) -> Dataset:
        return self.dataset

    def _generate_random_tokens(self, N_samples: int, n_ctx: int) -> np.ndarray:
        d_vocab = len(self.map_dict.keys())
        samples =  t.randint(2, d_vocab, (N_samples, n_ctx)).numpy() #remove BOS, PAD -- always assume these are the first 2 in the dictionary.
        return np.unique(samples, axis=0)
    
    def get_IIT_train_test_set(self, train_frac: float = 0.8, seed: int = 0) -> tuple[IITDataset, IITDataset]:

        decorated_dset = SimpleDataset(
            inputs = np.array(self.dataset['tokens']),
            targets = np.array(self.dataset['labels']),
            markers = np.array(self.dataset['markers'])
        )

        print("making IIT dataset")
        train_dataset, test_dataset = train_test_split(
            decorated_dset, test_size=1-train_frac, random_state=42
        )
        train_set = IITDataset(train_dataset, train_dataset, seed=seed)
        test_set = IITDataset(test_dataset, test_dataset, seed=seed)
        return train_set, test_set
    

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