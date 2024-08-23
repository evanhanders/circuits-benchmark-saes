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
        'a': 2, 
        'b': 3,
        'c': 4, 
        } 

REVERSE_CASE_VOCAB = {v: k for k, v in CASE_VOCAB.items()}

def create_duplicate_remover_tokenizer(verbose: bool = False) -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    hf_tokenizer = create_tokenizer(CASE_VOCAB)
    
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
        output[:, 0] = CASE_VOCAB['PAD']
        return output
    
class AreEqual(t.nn.Module):
    """ Checks equality of two tensors. """

    def forward(self, t1: Int[t.Tensor, "batch seq"], t2: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        return (t1 == t2).to(t.int)

class MaskedOutput(t.nn.Module):
    """ Masks an output tensor based on a boolean mask tensor. """
    def __init__(self, mask_token: int = CASE_VOCAB['PAD']):
        super().__init__()
        self.mask_token = mask_token

    def forward(self, input: Int[t.Tensor, "batch seq"], mask: Bool[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        output = input.clone()
        output[mask] = self.mask_token
        return output

class HighLevelDuplicateRemover(PolyCase):
    def __init__(self, vocab_dict: dict[str, int] = CASE_VOCAB, device: str = get_device()):
        super().__init__(vocab_dict=vocab_dict, device=device)
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
        return Correspondence(corr_node_dict, suffixes={'mlp': 'mlp.hook_post', 'attn': 'attn.hook_z'})

    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Tensor, t.Tensor]) -> Float[t.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)

        prev_tokens = self.prev_token_hook(self.previous_token_head(tokens))
        equal = self.prev_equal_hook(self.are_equal_head(tokens, prev_tokens))
        output = self.output_hook(self.masked_output(tokens, equal.to(bool)))

        # output pad at bos spot
        true_output = t.nn.functional.one_hot(output.long(), num_classes=self.d_vocab).float()
        
        return true_output.to(self.device)

    def __str__(self) -> str:
        return "duplicate_remover_model"

    
class DuplicateRemoverDataset(PolyBenchDataset):

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

    def _generate_token_subset(self, N_samples: int, n_ctx: int) -> np.ndarray:
        return self._generate_random_tokens(N_samples, n_ctx)

    def generate_tokens(self) -> None:

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        tokens = self._generate_token_subset(self.N_samples*2, self.n_ctx - 1)
        dataset = tokens[:self.N_samples]

        #add BOS token to beginning
        self.tokens = np.concatenate([
            self.map_dict['BOS']*np.ones((dataset.shape[0], 1)),
            dataset, 
        ], axis=1).astype(int)

    def generate_labels(self, skip_first: bool = False) -> None:
        hl_model = HighLevelDuplicateRemover()
        new_markers = np.zeros(self.tokens.shape, dtype=int)
        for i,sample in enumerate(self.tokens):
            if skip_first:
                sample = sample[1:]
            _, cache = hl_model.run_with_cache((t.tensor(sample).unsqueeze(0), None, None))
            if skip_first:
                new_markers[i][1:] = cache['output_hook']
            else:
                new_markers[i] = cache['output_hook']
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=len(self.map_dict.keys())).float().numpy() #-1 to remove UNK.