from abc import ABC, abstractmethod 
from jaxtyping import Float, Int, Bool
from typing import Optional

import torch as t
import numpy as np
from datasets import Dataset
from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import PreTrainedTokenizerFast
from iit.utils.correspondence import Correspondence, HLNode, LLNode
from iit.utils.index import Ix
from iit.utils.iit_dataset import train_test_split
from iit.utils.iit_dataset import IITDataset

from .utils import CustomDataset, create_tokenizer


PAREN_VOCAB = {
        0: '(', 
        1: ')', 
        2: ' PAD', 
        3: 'BOS', 
        4: ' UNK'
        } 

PAREN_REVERSE_VOCAB = {v: k for k, v in PAREN_VOCAB.items()}


def create_left_greater_tokenizer() -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    hf_tokenizer = create_tokenizer(PAREN_VOCAB)
    
    # Test the tokenizer
    encoded = hf_tokenizer.encode("BOS ( ) ( ) PAD PAD PAD")
    decoded = hf_tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    return hf_tokenizer

class LeftParenCountHead(t.nn.Module):
    """ Calculates how many left parens are in the series up to this token """

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        #tok_clone is 1 for ( and -1 for )
        tok_clone = tokens.clone()
        tok_clone[tok_clone == 0] = 300 #(
        tok_clone[tok_clone != 300] = 0 #), pad, bos
        tok_clone[tok_clone == 300] = 1 #(

        #we'll count left to right.
        lefts = t.cumsum(tok_clone, dim=1).to(int)

        return lefts

class RightParenCountHead(t.nn.Module):
    """ Calculates how many right parens are in the series up to this token """

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        #tok_clone is 1 for ( and -1 for )
        tok_clone = tokens.clone()
        tok_clone[tok_clone == 1] = 1 #)
        tok_clone[tok_clone != 1] = 0 #(, pad, bos

        #we'll count left to right.
        rights = t.cumsum(tok_clone, dim=1).to(int)

        return rights

class GreaterThan(t.nn.Module):
    """ Calculates if there are more left parens than right parens """
    
    def forward(self, lefts: Int[t.Tensor, "batch seq"], rights: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        return lefts > rights


class HighLevelLeftGreater(HookedRootModule):

    def __init__(self, device='cpu', d_vocab=4):
        super().__init__()
        self.d_vocab = d_vocab
        
        self.input_hook = HookPoint()
        self.left_parens = LeftParenCountHead()
        self.left_parens_hook = HookPoint()
        self.right_parens = RightParenCountHead()
        self.right_parens_hook = HookPoint()

        self.greater_than = GreaterThan()
        self.mlp0_hook = HookPoint()

        self.device = device
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
    
    def get_ll_model(self, cfg: Optional[HookedTransformerConfig] = None) -> HookedTransformer:
        if cfg is None:
            cfg = self.get_ll_model_cfg()
        return HookedTransformer(cfg)

    def get_correspondence(self) -> Correspondence:
        corr = {
            'input_hook' :           [('hook_embed', Ix[[None]],                None)],
            'left_parens_hook' :     [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
            'right_parens_hook' :    [('blocks.0.attn.hook_z',    Ix[[None, None, 1, None]], None)],
            'mlp0_hook':             [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
            'mlp1_hook' :            [('blocks.1.mlp.hook_post',  Ix[[None]], None)],
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

        left_parens = self.left_parens_hook(self.left_parens(tokens))
        right_parens = self.right_parens_hook(self.right_parens(tokens))

        greater_than = self.mlp0_hook(self.greater_than(left_parens, right_parens))
        
        # output pad at bos spot
        output = (greater_than).to(int)
        output[:,0] = 2
        true_output = t.nn.functional.one_hot(output, num_classes=self.d_vocab).float().to(self.device)
        
        return true_output



def test_HL_left_greater_components():
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
    true_mlp0_check = [ # left > right
        [ False,  True, False,  True, False,  True, False, False, False, False, False],
        [ False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [ False, False, False, False, False, False, False, False, False, False, False],
        [ False,  True, False,  True, False,  True, False,  True, False,  True, False],
        [ False, False, False, False, False, False, False, False, False, False, False],
    ]
    true_output  = [ # 2 in the first index, otherwise the integer version of true_mlp0_check
        [ 2, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [ 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_mlp0_check = t.Tensor(true_mlp0_check).to(bool)
    true_output = t.nn.functional.one_hot(t.Tensor(true_output).to(int), num_classes=4).float()

    checker = HighLevelLeftGreater()
    output, cache   = checker.run_with_cache((tokens, None, None))
    assert t.allclose(cache['left_parens_hook'], true_lefts)
    assert t.allclose(cache['right_parens_hook'], true_rights)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(output, true_output)
    print("All Balance tests passed!")

    return True

