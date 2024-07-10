from jaxtyping import Float, Int, Bool
from typing import Optional, Callable, Tuple

import torch as t
import numpy as np
from datasets import Dataset

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import PreTrainedTokenizerFast
from iit.utils.correspondence import Correspondence, HLNode, LLNode
from iit.utils.index import Ix

class ElevationHead(t.nn.Module):
    """ Calculates the elevation at each position in the context"""
    
    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        """ Vocabulary:
                0 - [START]
                1 - (
                2 - )
                3 - [PAD]
        """
        #tok_clone is 1 for ( and -1 for )
        tok_clone = tokens.clone()
        tok_clone[tok_clone == 3] = 0
        tok_clone[tok_clone == 2] = -1

        #we'll count left to right.
        elevation = t.cumsum(tok_clone, dim=1).to(int)

        return elevation

class CheckElevation(t.nn.Module):
    """ Checks if the elevation in token position -1 is 0.
        Returns 1 if so and 0 if false.
    """

    def forward(self, elevations: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        elevation_bool = t.ones(elevations.shape, dtype=t.bool)
        elevation_bool[elevations[:,-1].nonzero(), -1] = 0
        return elevation_bool

class CheckHorizon(t.nn.Module):
    """ Checks if the horizon is ever violated (elevation drops below 0)
    """

    def forward(self, elevations: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        horizon_bool = t.ones(elevations.shape, dtype=t.bool)
        horizon_bool[elevations < 0] = 0
        return horizon_bool

class BalanceCheckHead(t.nn.Module): #have this be a head that finds the minimum elevation.
    """ Checks to see if the balance has been violated previously in the sequence. """
    def forward(
        self, 
        horizon_check: Bool[t.Tensor, "batch seq"], 
        elevation_check: Bool[t.Tensor, "batch seq"]
    ) -> Bool[t.Tensor, "batch seq"]:
        both_checks = horizon_check*elevation_check
        return t.cumprod(both_checks, dim=1).bool() # Can the attention head do this, or can it just find out 

#TODO: add MLP operation for doing the product above (AND)
# MLP checks if min ele < 0 and whether elevation == 0

class HighLevelParensBalanceChecker(HookedRootModule):
    """
    Components:
    - Elevation Calculation Head
    - Elevation Check method
    - Horizon Check method
    - Balance Check Head
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.input_hook = HookPoint()
        self.elevation_head = ElevationHead()
        self.elevation_hook = HookPoint()
        self.elevation_checker = CheckElevation()
        self.elevation_check_hook = HookPoint()
        self.horizon_checker = CheckHorizon()
        self.horizon_check_hook = HookPoint()
        self.balance_check_head = BalanceCheckHead()
        self.balance_check_hook = HookPoint()
        self.device = device
        self.setup()

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Float[t.Tensor, "batch seq"]:
        tokens = self.input_hook(tokens)
        elevation = self.elevation_hook(self.elevation_head(tokens))
        ele_check = self.elevation_check_hook(self.elevation_checker(elevation))
        hor_check = self.horizon_check_hook(self.horizon_checker(elevation))
        balance = self.balance_check_hook(self.balance_check_head(hor_check, ele_check))
        return balance.float().to(self.device)

class ParensBalanceDataset:
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: Optional[int] = None
    ):
        # check if n_ctx is long enough for N_samples
        max_n_ctx = self.num_places(N_samples-1) + 1
        if n_ctx is not None and n_ctx < max_n_ctx:
            raise ValueError(f"specific n_ctx is too short for N_samples! Need n_ctx >= {max_n_ctx}.")

        self.N_samples = N_samples
        self.n_ctx = n_ctx if n_ctx is not None else max_n_ctx

        #generate lists of self.str_tokens and self.tokens:
        self.generate_tokens()

        #generate labels
        self.generate_labels()

        self.dataset = Dataset.from_dict({
                'tokens' : self.tokens,
                'str_tokens' : self.str_tokens,
                'labels': self.labels
            })

    def num_places(self, index: int):
        """ """
        n_places = 0
        while index >= 0:
            n_places += 1
            index -= 2**n_places
        return n_places
    
    def sample_parentheses(self, n: int):
        def to_binary(num: int, n_places: int):
            return f'{num:0{n_places}b}'
    
        def map_to_parentheses(binary_str):
            mapping = {'0': '(', '1': ')'}
            return ''.join(mapping[digit] for digit in binary_str)
    
        n_places = self.num_places(n)
        curr_number = n - (2**(n_places) - 2)
        binary_str = to_binary(curr_number, n_places)
        result = map_to_parentheses(binary_str)
        return result

    def generate_tokens(self):
        self.str_tokens = []
        self.tokens = []
        for i in range(self.N_samples):
            parentheses = self.sample_parentheses(i)
                
            # Using map with a lambda function
            binary_mapping = list(map(lambda x: 1 if x == '(' else 2, parentheses))
            n_pad = self.n_ctx - (len(binary_mapping) + 1)
            self.str_tokens.append(['[BOS]'] + list(parentheses) + n_pad*['[PAD]'])
            self.tokens.append([0] + binary_mapping + n_pad*[3,]) 
    
    def generate_labels(self):
        balance_checker = HighLevelParensBalanceChecker()
        inputs = t.Tensor(self.tokens).to(int)
        labels = balance_checker(inputs)
        self.labels = labels.int()[:,-1]

    def get_dataset(self):
        return self.dataset


def test_HL_parens_components():
    """ 
    Checks performance on a few pre-defined sequence inputs:
        1. [BOS] ( ) ( ) ( ) [PAD] [PAD] [PAD] [PAD]
        2. [BOS] ( ( ( ( ( ) ) ) [PAD] [PAD]
        3. [BOS] ) ( ) ( ) ( ) ( ) (
        4. [BOS] ( ) ( ) ( ) ( ) ( )
    """
    
    tokens = [
        [0, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3],
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3],
        [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    ]
    true_elevations = [
        [ 0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  1,  2,  3,  4,  5,  4,  3,  2,  2,  2],
        [ 0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0],
        [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0]
    ]
    true_ele_check = [
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True]
    ]
    true_hor_check = [
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False,  True, False,  True, False,  True, False,  True, False, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True]
    ]
    true_balance_check = [
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True]
    ]
    tokens = t.Tensor(tokens).to(int)
    true_elevations = t.Tensor(true_elevations).to(int)
    true_ele_check = t.Tensor(true_ele_check).to(bool)
    true_hor_check = t.Tensor(true_hor_check).to(bool)
    true_balance_check = t.Tensor(true_balance_check).to(bool)

    balance_checker = HighLevelParensBalanceChecker()
    balanced, cache = balance_checker.run_with_cache(tokens)
    assert t.allclose(balanced, true_balance_check.float())
    assert t.allclose(cache['elevation_hook'], true_elevations)
    assert t.allclose(cache['elevation_check_hook'], true_ele_check)
    assert t.allclose(cache['horizon_check_hook'], true_hor_check)
    assert t.allclose(cache['balance_check_hook'], true_balance_check)
    print("All tests passed!")

def get_LL_parens_model_and_correspondence(   
) -> Tuple[HookedTransformer, Correspondence, list]:
    """
    For the parenthesis task, returns:
    1. An initialized HookedTransformer for training
    2. The hand-specified correspondence between HL model hooks and LL model nodes for IIT
    3. A list of nodes in the LL model that are not used in the correspondence for SIIT
    """
    #Get HookedTransformer
    cfg = HookedTransformerConfig(
        n_layers = 2,
        d_model = 16,
        n_ctx = 20,
        d_head = 8,
        d_vocab = 4,
        act_fn = "relu",
    )
    model = HookedTransformer(cfg)

    #Get Correspondence
    # elevation and horizon check are mapped to the first and second half of MLP0 neurons.
    ele_neurons = t.arange(0,cfg.d_model*2).int()
    hor_neurons = t.arange(cfg.d_model*2, cfg.d_model*4).int()
    corr = {
        'input_hook' :           [('blocks.0.hook_resid_pre', Ix[[None]],                None)],
        'elevation_hook' :       [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
        'elevation_check_hook' : [('blocks.0.mlp.hook_post',  Ix[[None]],                ele_neurons)],
        'horizon_check_hook' :   [('blocks.0.mlp.hook_post',  Ix[[None]],                hor_neurons)], 
        'balance_check_hook' :   [('blocks.1.attn.hook_z',    Ix[[None, None, 1, None]], None)],
    }
    corr_node_dict = {}
    for hk, lks in corr.items():
        hn = HLNode(hk, -1)
        lns = {LLNode(name=k, index=idx, subspace=sp) for k, idx, sp in lks}
        corr_node_dict[hn] = lns
    corr_obj = Correspondence(corr_node_dict)

    #Get unused nodes
    #TODO: We could further restrict computation subspaces, and this code doesn't allow for that.
    unused_model_labels = [
        ('blocks.0.attn.hook_z', [1]), 
        ('blocks.1.attn.hook_z', [0]), 
        'blocks.1.mlp.hook_post'
    ]
    unused_hook_nodes = []
    for label in unused_model_labels:
        if isinstance(label, tuple):
            hook, heads = label
            for head in heads:
                unused_hook_nodes.append(LLNode(name=hook, index=Ix[[None,None,head,None]]))
        else:
            unused_hook_nodes.append(LLNode(name=label, index=Ix[[None]]))
            
    return model, corr_obj, unused_hook_nodes
    
def paren_checker_loss_fn(
    outputs : Float[t.Tensor, "batch n_ctx d_vocab"], #logits
    labels : Int[t.Tensor, "batch"]
) -> Float[t.Tensor, ""]:
    return t.nn.BCEWithLogitsLoss()(outputs[:,-1, -1], labels)

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
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'mask_token': '[MASK]'
    })
    return hf_tokenizer

def create_paren_checker_tokenizer() -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    vocab = {'BOS': 0, '(': 1, ')': 2, '[PAD]': 3, 'UNK' : 4}
    hf_tokenizer = create_tokenizer(vocab)
    
    # Test the tokenizer
    encoded = hf_tokenizer.encode("BOS ( ) ( ) [PAD] [PAD] [PAD]")
    decoded = hf_tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    return hf_tokenizer