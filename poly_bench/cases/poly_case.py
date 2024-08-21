from abc import ABC, abstractmethod
from typing import Optional

import torch as t
import numpy as np
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookedRootModule
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import get_device
from datasets import Dataset

from iit.utils.iit_dataset import train_test_split, IITDataset
from iit.utils.correspondence import Correspondence

from .utils import CustomDataset


class PolyCase(HookedRootModule, ABC):
    def __init__(self, vocab_dict: dict[str, int], device: str = get_device()):
        super().__init__()
        self.vocab_dict = vocab_dict
        self.d_vocab = len(vocab_dict.keys()) - 1 #-1 because we don't do computations for UNK token
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
        n_ctx: Optional[int] = None,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.N_samples = N_samples
        self.n_ctx = n_ctx
        if map_dict is None:
            raise ValueError("map_dict must be provided")
        self.map_dict = map_dict
        self.reverse_map_dict: dict[str, int] = {v: k for k, v in map_dict.items()}

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
    def generate_tokens(self):
        pass

    def map_tokens_to_str(self):
        # Vectorized mapping using numpy
        vectorized_map = np.vectorize(self.map_dict.get)
        self.str_tokens = vectorized_map(self.tokens)

    @abstractmethod
    def generate_labels(self, skip_first: bool = False) -> None:
        pass

    def get_dataset(self):
        return self.dataset

    def _generate_random_tokens(self, N_samples, n_ctx):
        d_vocab = len(self.map_dict.keys()) - 3 #remove BOS, PAD, UNK -- always assume these are the last 3 in the dictionary.
        samples =  t.randint(0, d_vocab, (N_samples, n_ctx))
        return t.unique(samples, dim=0)
    
    def get_IIT_train_test_set(self, train_frac=0.8, seed=0):

        decorated_dset = CustomDataset(
            inputs = self.dataset['tokens'],
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
        
    def left_greater(self,  sample):
        return np.cumsum(sample == 0) > np.cumsum(sample == 1)