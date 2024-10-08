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

    
class AppearedPreviously(t.nn.Module):
    """ Checks equality of two tensors. """

    def forward(self, 
                tokens: Int[t.Tensor, "batch seq"],
                token_counts: dict[int, Int[t.Tensor, "batch seq"]]
                ) -> dict[int, Int[t.Tensor, "batch seq"]]:
    
        output = {}
        for token, count in token_counts.items():
            output[token] = t.logical_and(count > 1, tokens == token).to(t.int)
        return output

class MaskBuilder(t.nn.Module):
    """ Given a list of torch tensors of bools, logical_ors them together. """

    def forward(self, masks: list[Bool[t.Tensor, "batch seq"]]) -> Int[t.Tensor, "batch seq"]:
        output = t.zeros_like(masks[0])
        for mask in masks:
            output = t.logical_or(output, mask)
        return output.to(t.int)

class MaskedOutput(t.nn.Module):
    """ Masks an output tensor based on a boolean mask tensor. """
    def __init__(self, mask_token: int = CASE_VOCAB['PAD']):
        super().__init__()
        self.mask_token = mask_token

    def forward(self, input: Int[t.Tensor, "batch seq"], mask: Bool[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        output = input.clone()
        output[mask] = self.mask_token
        return output

class HighLevelUniqueExtractor(PolyCase):
    def __init__(self, vocab_dict: dict[str, int] = CASE_VOCAB, device: str = get_device()):
        super().__init__(vocab_dict=vocab_dict, device=device)
        self.input_hook = HookPoint()
        self.counter_head = HookPoint()
        self.appeared_mlp = HookPoint()
        self.mask_mlp = HookPoint()
        self.output_mlp = HookPoint()

        self.a_token_count = TokenCountHead(self.vocab_dict['a'])
        self.b_token_count = TokenCountHead(self.vocab_dict['b'])
        self.c_token_count = TokenCountHead(self.vocab_dict['c'])
        self.appeared = AppearedPreviously()
        self.mask_builder = MaskBuilder()
        self.mask_output = MaskedOutput()

        self.setup()

    def get_ll_model_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers = 3,
            d_model = 32,
            n_ctx = 15,
            d_head = 8,
            d_vocab = self.d_vocab,
            act_fn = "relu"
        )

    def get_correspondence(self) -> Correspondence:
        corr = {
            'input_hook' :           [('hook_embed', Ix[[None]], None)],
            'counter_head' :         [('blocks.0.attn.hook_z',    Ix[[None, None, 2, None]], None)],
            'appeared_mlp' :         [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
            'mask_mlp' :             [('blocks.1.mlp.hook_post',  Ix[[None]], None)],
            'output_mlp' :           [('blocks.2.mlp.hook_post',  Ix[[None]], None)],
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

        a_count = self.a_token_count(tokens)
        b_count = self.b_token_count(tokens)
        c_count = self.c_token_count(tokens)
        counts = self.counter_head(t.stack([a_count, b_count, c_count]))
        a_count = counts[0]
        b_count = counts[1]
        c_count = counts[2]

        input_to_appeared = {
            self.vocab_dict['a']: a_count,
            self.vocab_dict['b']: b_count,
            self.vocab_dict['c']: c_count
        }
        appeared = self.appeared(tokens, input_to_appeared)
        appeared_hook = self.appeared_mlp(t.stack( [
            appeared[self.vocab_dict['a']], 
            appeared[self.vocab_dict['b']], 
            appeared[self.vocab_dict['c']]
            ]))

        mask = self.mask_builder([appeared_hook[0].to(bool), appeared_hook[1].to(bool), appeared_hook[2].to(bool)])
        mask = self.mask_mlp(mask)

        output = self.output_mlp(self.mask_output(tokens, mask.to(bool)))
        # output pad at bos spot
        true_output = t.nn.functional.one_hot(output.long(), num_classes=self.d_vocab).float()
        
        return true_output.to(self.device)

    def __str__(self) -> str:
        return "unique_extractor_model"

class UniqueExtractorDataset(PolyBenchDataset):

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
        hl_model = HighLevelUniqueExtractor()
        new_markers = np.zeros(self.tokens.shape, dtype=int)
        for i,sample in enumerate(self.tokens):
            if skip_first:
                sample = sample[1:]
            _, cache = hl_model.run_with_cache((t.tensor(sample).unsqueeze(0), None, None))
            if skip_first:
                new_markers[i][1:] = cache['output_mlp']
                new_markers[i][0] = 0
            else:
                new_markers[i] = cache['output_mlp']
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=len(self.map_dict.keys())).float().numpy() #-1 to remove UNK.
