from abc import ABC, abstractmethod 
from jaxtyping import Float, Int, Bool
from typing import Optional

import torch as t
import numpy as np
from datasets import Dataset

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import PreTrainedTokenizerFast
from iit.utils.correspondence import Correspondence, HLNode, LLNode
from iit.utils.index import Ix
from iit.utils.iit_dataset import train_test_split
from iit.utils.iit_dataset import IITDataset

from .utils import CustomDataset, create_tokenizer

CASE_VOCAB = {
        0: 'a', 
        1: 'b',
        2: 'c', 
        3: 'PAD', 
        4: 'BOS', 
        5: 'UNK'
        } 

REVERSE_CASE_VOCAB = {v: k for k, v in CASE_VOCAB.items()}

def create_duplicate_remover_tokenizer(verbose: bool = False) -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    hf_tokenizer = create_tokenizer(REVERSE_CASE_VOCAB)
    
    # Test the tokenizer
    if verbose:
        encoded = hf_tokenizer.encode("BOS a a b c a b PAD PAD")
        decoded = hf_tokenizer.decode(encoded)
        print("Tokenizer test:")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
    return hf_tokenizer

class PreviousTokenHead(t.nn.Module):
    """ Outputs the token before this token. """

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        output = t.zeros_like(tokens)
        output[:, 1:] = tokens[:, :-1]
        output[:, 0] = REVERSE_CASE_VOCAB['PAD']
        return output
    
class AreEqual(t.nn.Module):
    """ Checks equality of two tensors. """

    def forward(self, t1: Int[t.Tensor, "batch seq"], t2: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        return (t1 == t2).to(int)

class MaskedOutput(t.nn.Module):
    """ Masks an output tensor based on a boolean mask tensor. """
    def __init__(self, mask_token: int = REVERSE_CASE_VOCAB['PAD']):
        super().__init__()
        self.mask_token = mask_token

    def forward(self, input: Int[t.Tensor, "batch seq"], mask: Bool[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        output = input.clone()
        output[mask] = self.mask_token
        return output

class HighLevelDuplicateRemover(HookedRootModule):
    def __init__(self, d_vocab=5):
        super().__init__()
        self.d_vocab = d_vocab
        self.input_hook = HookPoint()
        self.previous_token_head = PreviousTokenHead()
        self.prev_token_hook = HookPoint()
        self.are_equal_head = AreEqual()
        self.prev_equal_hook = HookPoint()
        self.masked_output = MaskedOutput()
        self.output_hook = HookPoint()

        self.setup()

    def get_ll_model_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers = 2,
            d_model = 20,
            n_ctx = 15,
            d_head = 5,
            d_vocab = self.d_vocab,
            act_fn = "relu"
        )
    
    def get_ll_model(self, cfg: Optional[HookedTransformerConfig] = None) -> HookedTransformer:
        if cfg is None:
            cfg = self.get_ll_model_cfg()
        return HookedTransformer(cfg)

    def get_correspondence(self) -> Correspondence:
        corr = {
            'input_hook' :           [('hook_embed', Ix[[None]], None)],
            'prev_token_hook' :      [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
            'prev_equal_hook':       [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
            'output_hook' :          [('blocks.1.mlp.hook_post',  Ix[[None]], None)],
        }
        corr_node_dict = {}
        for hk, lks in corr.items():
            hn = HLNode(hk, -1)
            lns = {LLNode(name=k, index=idx, subspace=sp) for k, idx, sp in lks}
            corr_node_dict[hn] = lns
        return Correspondence(corr_node_dict)
         
    def is_categorical(self) -> bool:
        return True

    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Tensor, t.Tensor]) -> Float[t.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)

        prev_tokens = self.prev_token_hook(self.previous_token_head(tokens))
        equal = self.prev_equal_hook(self.are_equal_head(tokens, prev_tokens))
        output = self.output_hook(self.masked_output(tokens, equal.to(bool)))

        # output pad at bos spot
        true_output = t.nn.functional.one_hot(output, num_classes=self.d_vocab).float()
        
        return true_output
    


def test_HL_duplicate_remover_components():
    # parens balance check
    tokens = [
        "BOS a a b c a b PAD PAD",
        "BOS a b c c c c c c",
        "BOS a b c PAD PAD PAD PAD PAD",
    ]
    tokenizer = create_duplicate_remover_tokenizer()
    encoded = [tokenizer.encode(t) for t in tokens]
    print(encoded)
    true_prev_tokens = [[REVERSE_CASE_VOCAB['PAD']] + e[:-1] for e in encoded]
    true_equal = [[a == b for a, b in zip(e, p)] for e, p in zip(encoded, true_prev_tokens)]
    print(true_equal)
    true_output = [[REVERSE_CASE_VOCAB['PAD'] if eq else a for a, eq in zip(encoded[i], true_equal[i])] for i in range(len(tokens))]

    tokens = t.Tensor(encoded).to(int)
    true_prev_tokens = t.Tensor(true_prev_tokens).to(int)
    true_equal = t.Tensor(true_equal).to(int)
    true_output = t.Tensor(true_output).to(int)
    
    checker = HighLevelDuplicateRemover()
    _, cache   = checker.run_with_cache((tokens, None, None))
    # print(cache['right_parens_hook'] - true_rights)
    assert t.allclose(cache['prev_token_hook'], true_prev_tokens)
    assert t.allclose(cache['prev_equal_hook'], true_equal)
    assert t.allclose(cache['output_hook'], true_output)
    print("All DuplicateRemover tests passed!")

    return True



class PolyBenchDataset(ABC):
    
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: Optional[int] = None,
        seed: int = 42,
        map_dict: dict[int, str] = CASE_VOCAB,
    ):
        np.random.seed(seed)
        self.N_samples = N_samples
        self.n_ctx = n_ctx
        self.map_dict = map_dict
        self.reverse_map_dict: dict[str, int] = {v: k for k, v in map_dict.items()}

        #generate lists of self.str_tokens and self.tokens:
        self.generate_tokens()
        self.map_tokens_to_str()

        #generate labels
        self.generate_labels()

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
    def generate_labels(self):
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
    
    
class DuplicateRemoverDataset(PolyBenchDataset):

    def _generate_token_subset(self, N_samples, n_ctx):
        return self._generate_random_tokens(N_samples, n_ctx)

    def generate_tokens(self):

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        tokens = self._generate_token_subset(self.N_samples*2, self.n_ctx - 1)
        dataset = tokens[:self.N_samples]

        #add BOS token to beginning
        self.tokens = np.concatenate([
            self.reverse_map_dict['BOS']*np.ones((dataset.shape[0], 1)),
            dataset, 
        ], axis=1).astype(int)

    def generate_labels(self):
        hl_model = HighLevelDuplicateRemover()
        new_markers = np.zeros(self.tokens.shape, dtype=int)
        for i,sample in enumerate(self.tokens):
            _, cache = hl_model.run_with_cache((t.tensor(sample).unsqueeze(0), None, None))
            new_markers[i] = cache['output_hook']
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=len(self.map_dict.keys()) - 1).float().numpy() #-1 to remove UNK.
        print(self.labels.shape)

    
