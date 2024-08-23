from jaxtyping import Float, Int, Bool
from typing import Optional

import torch as t
import numpy as np

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig # type: ignore
from transformer_lens.hook_points import HookPoint # type: ignore
from transformer_lens.utils import get_device # type: ignore
from transformers import PreTrainedTokenizerFast # type: ignore
from iit.utils.correspondence import Correspondence, HLNode, LLNode # type: ignore
from iit.utils.index import Ix # type: ignore


from .poly_case import PolyCase, PolyBenchDataset, create_tokenizer


CASE_VOCAB = {
        'BOS': 0, 
        'PAD': 1, 
        '(': 2, 
        ')': 3, 
        } 

CASE_REVERSE_VOCAB = {v: k for k, v in CASE_VOCAB.items()}


def create_left_greater_tokenizer() -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    hf_tokenizer = create_tokenizer(CASE_VOCAB)
    
    # Test the tokenizer
    encoded = hf_tokenizer.encode("BOS ( ) ( ) PAD PAD PAD")
    decoded = hf_tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    return hf_tokenizer


class TokenCountHead(t.nn.Module):
    """ Counts the number of tokens in the series up to this token """
    def __init__(self, token_to_count: int):
        super().__init__()
        self.token_to_count = token_to_count

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        tok_clone = tokens.clone()
        tok_clone[tok_clone == self.token_to_count] = -1
        tok_clone[tok_clone != -1] = 0
        tok_clone[tok_clone == -1] = 1
        return t.cumsum(tok_clone, dim=1).to(t.int)

class GreaterThan(t.nn.Module):
    """ Calculates if there are more left parens than right parens """
    
    def forward(self, lefts: Int[t.Tensor, "batch seq"], rights: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        return lefts > rights


class HighLevelLeftGreater(PolyCase):

    def __init__(self, vocab_dict: dict[str, int] = CASE_VOCAB, device: str = get_device()):
        super().__init__(vocab_dict=vocab_dict, device=device)
        
        self.input_hook = HookPoint()
        self.left_parens = TokenCountHead(self.vocab_dict['('])
        self.right_parens =TokenCountHead(self.vocab_dict[')'])
        self.paren_counts_hook = HookPoint()

        self.greater_than = GreaterThan()
        self.mlp0_hook = HookPoint()

        self.setup()
    
    def get_ll_model_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers = 1,
            d_model = 12,
            n_ctx = 15,
            d_head = 3,
            d_vocab = self.d_vocab,
            act_fn = "relu"
        )

    def get_correspondence(self) -> Correspondence:
        corr = {
            'input_hook' :           [('hook_embed', Ix[[None]],                None)],
            'paren_counts_hook' :    [('blocks.0.attn.hook_z',    Ix[[None, None, 1, None]], None)],
            'mlp0_hook':             [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
        }
        corr_node_dict = {}
        for hk, lks in corr.items():
            hn = HLNode(hk, -1)
            lns = {LLNode(name=k, index=idx, subspace=sp) for k, idx, sp in lks}
            corr_node_dict[hn] = lns
        return Correspondence(corr_node_dict, suffixes={'mlp': 'mlp.hook_post', 'attn': 'attn.hook_z'})
         
    def is_categorical(self) -> bool:
        return True

    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Tensor, t.Tensor]) -> Float[t.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)

        left_parens = self.left_parens(tokens)
        right_parens = self.right_parens(tokens)
        parens = self.paren_counts_hook(t.stack([left_parens, right_parens]))
        left_parens = parens[0]
        right_parens = parens[1]

        greater_than = self.mlp0_hook(self.greater_than(left_parens, right_parens))
        
        # output 2 at bos spot
        output = (greater_than).to(int)
        output[:,0] = 2
        true_output = t.nn.functional.one_hot(output, num_classes=self.d_vocab).float().to(self.device)
        
        return true_output

    def __str__(self) -> str:
        return "left_greater_model"

class LeftGreaterDataset(PolyBenchDataset):
    
    def __init__(
        self, 
        N_samples: int, 
        map_dict: Optional[dict[str, int]] = CASE_VOCAB,
        n_ctx: int = 15,
        seed: int = 42,
    ):
        super().__init__(
            N_samples=N_samples, 
            map_dict=map_dict, 
            n_ctx=n_ctx, 
            seed=seed
            )
        
    def left_greater(self, sample: np.ndarray) -> np.ndarray:
        return np.cumsum(sample == self.map_dict['(']) > np.cumsum(sample == self.map_dict[')'])

    def _generate_token_subset(self, N_samples: int, n_ctx: int, left_greater: bool = True) -> np.ndarray:        
        generated_samples = min(N_samples, 1000)
        remaining_samples = N_samples
        good_samples = []
        while remaining_samples > 0:
            if left_greater:
                pos_lengths = t.randint(n_ctx//2+1, n_ctx, (generated_samples,))
            else:
                pos_lengths = t.randint(0, n_ctx//2, (generated_samples,))
            neg_lengths = n_ctx - pos_lengths

            samples = [ t.cat((
                t.ones((p.item())),
                -t.ones((n.item()))
            )) for p, n in zip(pos_lengths, neg_lengths)]
            stacked_samples = t.stack(samples)
            
            # shuffle
            indices = t.stack([t.randperm(n_ctx) for _ in range(generated_samples)])
            shuffled = t.gather(stacked_samples, 1, indices)
            good_samples.append(shuffled)
            remaining_samples -= good_samples[-1].shape[0]
            
        return t.unique(t.cat(good_samples, dim=0), dim=0).numpy()
        
    def _generate_left_greater(self, N_samples: int, n_ctx: int) -> np.ndarray:
        return self._generate_token_subset(N_samples, n_ctx, left_greater=True)
    
    def _generate_left_not_greater(self, N_samples: int, n_ctx: int) -> np.ndarray:
        return self._generate_token_subset(N_samples, n_ctx, left_greater=False)

    def generate_tokens(self) -> None:

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        greater = self._generate_token_subset(self.N_samples, self.n_ctx - 1, left_greater=True)
        less = self._generate_token_subset(self.N_samples, self.n_ctx - 1, left_greater=False)

        dataset = np.concatenate([
            greater[:self.N_samples // 2],
            less[:self.N_samples // 2],
        ], axis = 0)
        dataset[dataset == 1]  = self.map_dict['(']
        dataset[dataset == -1] = self.map_dict[')']
        dataset = dataset[t.randperm(dataset.shape[0]),:] #shuffle the dataset.

        #add BOS token to beginning and pad to end
        self.tokens = np.concatenate([
            self.map_dict['BOS']*np.ones((dataset.shape[0], 1)), #BOS
            dataset,
        ], axis=1).astype(int)

    def generate_labels(self, skip_first: bool = False) -> None:
        new_markers = np.zeros(self.tokens.shape, dtype=int)
        for i,sample in enumerate(self.tokens):
            if skip_first:
                sample = sample[2:]
            else:
                sample = sample[1:]
            mask = self.left_greater(sample)
            if skip_first:
                new_markers[i,2:][mask] = 1
            else:
                new_markers[i,1:][mask] = 1
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        if skip_first:
            self.labels[:,1] = 2 #set 2 as answer for bos token.
        else:
            self.labels[:,0] = 2 #set 2 as answer for bos token.
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=len(self.map_dict.keys())).float().numpy()

