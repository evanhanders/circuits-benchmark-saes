from abc import ABC, abstractmethod 
from jaxtyping import Float, Int, Bool
from typing import Optional

import torch as t
import numpy as np
from datasets import Dataset

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utils import get_device
from transformers import PreTrainedTokenizerFast
from iit.utils.correspondence import Correspondence, HLNode, LLNode
from iit.utils.index import Ix

from .utils import create_tokenizer
from .poly_case import PolyCase, PolyBenchDataset


CASE_VOCAB = {
        '(': 0, 
        ')': 1, 
        'PAD': 2, 
        'BOS': 3, 
        'UNK': 4
        } 

CASE_REVERSE_VOCAB = {v: k for k, v in CASE_VOCAB.items()}

def create_paren_checker_tokenizer() -> PreTrainedTokenizerFast:
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
        return t.cumsum(tok_clone, dim=1).to(int)

class ElevationCalculator(t.nn.Module):
    """ Calculates the elevation at each position in the context"""
    
    def forward(self, lefts: Int[t.Tensor, "batch seq"], rights: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        return lefts - rights

class CheckElevation(t.nn.Module):
    """ Checks if the elevation in token position -1 is 0.
        Returns 1 if so and 0 if false.
    """

    def forward(self, elevations: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        elevation_bool = t.ones(elevations.shape, dtype=t.bool)
        elevation_bool[elevations != 0] = 0
        return elevation_bool

class CheckHorizon(t.nn.Module):
    """ Checks if the horizon is ever violated (elevation drops below 0)
    """

    def forward(self, elevations: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        horizon_bool = t.ones(elevations.shape, dtype=t.bool)
        horizon_bool[elevations < 0] = 0
        return horizon_bool

class HorizonLookbackHead(t.nn.Module): #have this be a head that finds the minimum elevation.
    """ Checks to see if the horizon has been violated at any point in the sequence. """
    def forward(
        self, 
        horizon_check: Bool[t.Tensor, "batch seq"]
    ) -> Bool[t.Tensor, "batch seq"]:
        #this basically just gives True if horizon has never been violated before or now; false if horizon has been violated
        return t.cumprod(horizon_check, dim=1).bool()

class HighLevelParensBalanceChecker(PolyCase):

    def __init__(self, vocab_dict: Optional[dict[str, int]] = CASE_VOCAB, device=get_device()):
        super().__init__(vocab_dict, device=device)
        
        self.input_hook = HookPoint()
        self.left_parens = TokenCountHead(self.vocab_dict['('])
        self.right_parens = TokenCountHead(self.vocab_dict[')'])
        self.paren_counts_hook = HookPoint()

        self.elevation_calc = ElevationCalculator()
        self.elevation_hook = HookPoint()
        self.mlp0_hook = HookPoint()

        self.elevation_checker = CheckElevation()
        self.horizon_checker = CheckHorizon()
        self.mlp1_hook = HookPoint()

        self.mlp2_hook = HookPoint()

        self.horizon_lookback_head = HorizonLookbackHead()
        self.horizon_lookback_hook = HookPoint()

        self.setup()
    
    def get_ll_model_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers = 3,
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
            'input_hook' :           [('hook_embed', Ix[[None]],                None)],
            'paren_counts_hook' :    [('blocks.0.attn.hook_z',    Ix[[None, None, 3, None]], None)],
            'mlp0_hook':             [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
            'mlp1_hook' :            [('blocks.1.mlp.hook_post',  Ix[[None]], None)],
            'horizon_lookback_hook': [('blocks.2.attn.hook_z',    Ix[[None, None, 3, None]], None)],
            'mlp2_hook' :            [('blocks.2.mlp.hook_post',  Ix[[None]], None)]
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
        parens_counts = self.paren_counts_hook(t.stack((left_parens, right_parens)))
        left_parens = parens_counts[0]
        right_parens = parens_counts[1]

        elevation = self.elevation_hook(self.elevation_calc(left_parens, right_parens))
        elevation = self.mlp0_hook(elevation)
        
        ele_check = self.elevation_checker(elevation)
        hor_check = self.horizon_checker(elevation)
        hook_mlp1 = self.mlp1_hook(t.stack((ele_check, hor_check)))
        ele_check = hook_mlp1[0]
        hor_check = hook_mlp1[1]
        hor_lookback = self.horizon_lookback_hook(self.horizon_lookback_head(hor_check))

        output = (ele_check*hor_lookback).to(int)
        output[:,0] = 2
        output = self.mlp2_hook(output)

        # output pad at bos spot
        true_output = t.nn.functional.one_hot(output, num_classes=self.d_vocab).float().to(self.device)
        
        return true_output


class BalancedParensDataset(PolyBenchDataset):
    def __init__(
        self, 
        N_samples: int, 
        map_dict: Optional[dict[str, int]] = CASE_VOCAB,
        n_ctx: Optional[int] = 40,
        seed: int = 42,
    ):
        super().__init__(
            N_samples=N_samples, 
            map_dict=map_dict, 
            n_ctx=n_ctx, 
            seed=seed
            )

    def passes_balance(self,  sample):
        return np.cumsum(sample == 0) == np.cumsum(sample == 1)
        
    def passes_horizon(self, sample):
        mod = np.copy(sample)
        mod[mod == 1] = -1
        mod[mod == 0] = 1
        mod[mod > 1] = 0
        horizon = np.cumsum(mod)
        horizon_bool = np.ones(horizon.shape, dtype=bool)
        horizon_bool[horizon < 0] = 0
        horizon_lookback = np.cumprod(horizon_bool)
        return horizon_lookback.astype(bool)
    
    def passes_balance_test(self, sample):
        return self.passes_balance(sample)*self.passes_horizon(sample)

    def _generate_token_subset(self, N_samples, n_ctx, passes_balance=True, passes_horizon=True):
        """ Samples that fail both tests """
        assert n_ctx % 2 == 0, "n_ctx must be even."
        
        generated_samples = min(N_samples, 1000)
        remaining_samples = N_samples
        good_samples = []
        while remaining_samples > 0:
            if passes_balance:
                pos_lengths = (n_ctx//2) * t.ones(generated_samples).to(int)
                neg_lengths = pos_lengths
            else:
                # Generate +1 and -1 tensors so we ensure imbalance
                # There's def a smarter way to do this with just generating the ones and then indexing.
                coin_flip = t.randint(0, 2, (generated_samples,))
                #Generate all from 0 - n_ctx//2 - 1
                pos_lengths = t.randint(0, n_ctx//2, (generated_samples,))
                #use coin flip to flip half to range n_ctx//2 + 1 -> n_ctx
                pos_lengths[coin_flip == 0] = t.randint(n_ctx//2 + 1, n_ctx, ((coin_flip == 0).sum(),))
                neg_lengths = n_ctx - pos_lengths
            samples = [ t.cat((
                t.ones((p.item())),
                -t.ones((n.item()))
            )) for p, n in zip(pos_lengths, neg_lengths)]
            samples = t.stack(samples)
            
            # shuffle
            indices = t.stack([t.randperm(n_ctx) for _ in range(generated_samples)])
            shuffled = t.gather(samples, 1, indices)

            # create random elevations
            elevation = t.cumsum(shuffled, dim=1)
            min_values, _ = t.min(elevation, dim=1)
            
            # Step 6: Create appropriate mask for pass/fail on task
            if passes_horizon:
                mask = min_values >= 0
            else:
                mask = min_values < 0
            masked_samples = shuffled[mask,:]
            good_samples.append(masked_samples[:remaining_samples])
            remaining_samples -= good_samples[-1].shape[0]
            
        return t.unique(t.cat(good_samples, dim=0), dim=0)
        
    def _generate_balanced_tokens(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=True, passes_horizon=True)

    def _generate_horizon_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=True, passes_horizon=False)
        
    def _generate_balance_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=False, passes_horizon=True)
        
    def _generate_absolute_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=False, passes_horizon=False)
        
    def generate_tokens(self):

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        balanced = self._generate_balanced_tokens(self.N_samples, self.n_ctx - 1)
        fail_ele = self._generate_horizon_failures(self.N_samples // 2, self.n_ctx - 1)
        fail_bal = self._generate_balance_failures(self.N_samples // 2, self.n_ctx - 1)
        fail_both = self._generate_absolute_failures(self.N_samples // 2, self.n_ctx - 1)

        dataset = t.cat([
            balanced[:self.N_samples // 4],
            fail_ele[:self.N_samples // 4],
            fail_bal[:self.N_samples // 4],
            fail_both[:self.N_samples // 4]
        ], dim = 0)
        dataset[dataset == 1]  = 0 #(
        dataset[dataset == -1] = 1 #)
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
            new_markers[i,1:][self.passes_balance_test(sample)] = 1
            new_markers[i,1:][np.logical_and(~self.passes_balance(sample), self.passes_horizon(sample))] = 2
            new_markers[i,1:][np.logical_and(~self.passes_horizon(sample), self.passes_balance(sample))] = 3
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels[self.labels != 1] = 0 #passes or fails.
        self.labels[:,0] = 2 #set pad as answer for bos token.
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=4).float().numpy()


def test_HL_parens_balancer_components():
    # parens balance check
    tokens = [
        [3, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [3, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        [3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [3, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    ]
    true_lefts = [
        [ 0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3],
    ]
    true_rights = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7],
    ]
    true_mlp0_check = [ #elevations
        [ 0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  1,  2,  3,  4,  5,  4,  3,  2,  2,  2],
        [ 0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0],
        [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [ 0, -1, -2, -1, -2, -3, -2, -3, -4, -3, -4],
    ]
    horizon_check = [ #first element is ele; rest are horizon
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False,  True, False,  True, False,  True, False,  True, False, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
    ]
    elevation_check = [ #True where mlp0_check is 0, false otherwise.
        [ True, False,  True, False,  True, False,  True,  True,  True,  True,  True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True, False,  True, False,  True, False,  True, False,  True, False,  True],
        [ True, False,  True, False,  True, False,  True, False,  True, False,  True],
        [ True, False, False, False, False, False, False, False, False, False, False],
    ]
    true_mlp1_check = [
        elevation_check, horizon_check
    ]
    true_hor_lookback = [ # For each example, this is False as soon as horizon_check is false once.
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        ]
    true_output  = [ # boolean product of true_hor_lookback and elevation_check
        [ 2, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_parens_counts = t.stack((true_lefts, true_rights))
    true_mlp0_check = t.Tensor(true_mlp0_check).to(int)
    true_mlp1_check = t.Tensor(true_mlp1_check).to(bool)
    true_hor_lookback = t.Tensor(true_hor_lookback).to(bool)
    true_output = t.Tensor(true_output).to(int)
    

    checker = HighLevelParensBalanceChecker()
    _, cache   = checker.run_with_cache((tokens, None, None))
    # print(cache['right_parens_hook'] - true_rights)
    assert t.allclose(cache['paren_counts_hook'], true_parens_counts)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(cache['mlp1_hook'], true_mlp1_check)
    assert t.allclose(cache['horizon_lookback_hook'], true_hor_lookback)
    assert t.allclose(cache['mlp2_hook'], true_output)
    print("All Balance tests passed!")

    return True
