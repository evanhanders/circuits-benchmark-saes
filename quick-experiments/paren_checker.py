from abc import ABC, abstractmethod 
from jaxtyping import Float, Int, Bool
from typing import Optional, Callable, Tuple

import torch as t
import numpy as np
from datasets import Dataset
from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import PreTrainedTokenizerFast
from iit.utils.correspondence import Correspondence, HLNode, LLNode
from iit.utils.index import Ix

class LeftParenCountHead(t.nn.Module):
    """ Calculates how many left parens are in the series up to this token """

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        """ Vocabulary:
                0 - [START]
                1 - (
                2 - )
                3 - [PAD]
        """
        #tok_clone is 1 for ( and -1 for )
        tok_clone = tokens.clone()
        tok_clone[tok_clone == 0] = 5 #(
        tok_clone[tok_clone != 5] = 0 #), pad, bos
        tok_clone[tok_clone == 5] = 1 #(

        #we'll count left to right.
        lefts = t.cumsum(tok_clone, dim=1).to(int)

        return lefts

class RightParenCountHead(t.nn.Module):
    """ Calculates how many right parens are in the series up to this token """

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        """ Vocabulary:
                0 - [START]
                1 - (
                2 - )
                3 - [PAD]
        """
        #tok_clone is 1 for ( and -1 for )
        tok_clone = tokens.clone()
        tok_clone[tok_clone == 1] = 1 #)
        tok_clone[tok_clone != 1] = 0 #(, pad, bos

        #we'll count left to right.
        rights = t.cumsum(tok_clone, dim=1).to(int)

        return rights


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
        elevation_bool[elevations[:,-1].nonzero(), -1] = 0
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

class BalanceCheck(t.nn.Module):
    """ Performs AND to see if balance and horizon both pass """
    def forward(
        self,
        horizon_lookback : Bool[t.Tensor, "batch seq"],
        elevation_check : Bool[t.Tensor, "batch seq"]
    ):
        return horizon_lookback * elevation_check

class HighLevelParensBalanceChecker(HookedRootModule):
    """
    Components:
    - Elevation Calculation Head
    - Elevation Check method
    - Horizon Check method
    - Horizon lookback Head
    - Balance Check method
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.input_hook = HookPoint()
        self.left_paren_head = LeftParenCountHead()
        self.left_paren_hook = HookPoint()
        self.right_paren_head = RightParenCountHead()
        self.right_paren_hook = HookPoint()
        self.elevation_calc = ElevationCalculator()
        self.elevation_hook = HookPoint()
        self.elevation_checker = CheckElevation()
        self.elevation_check_hook = HookPoint()
        self.horizon_checker = CheckHorizon()
        self.horizon_check_hook = HookPoint()
        self.horizon_lookback_head = HorizonLookbackHead()
        self.horizon_lookback_hook = HookPoint()
        self.balance_check = BalanceCheck()
        self.balance_check_hook = HookPoint()
        self.device = device
        self.setup()

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Float[t.Tensor, "batch seq"]:
        tokens = self.input_hook(tokens)
        lefts = self.left_paren_hook(self.left_paren_head(tokens))
        rights = self.right_paren_hook(self.right_paren_head(tokens))
        elevation = self.elevation_hook(self.elevation_calc(lefts, rights))
        ele_check = self.elevation_check_hook(self.elevation_checker(elevation))
        hor_check = self.horizon_check_hook(self.horizon_checker(elevation))
        hor_lookback = self.horizon_lookback_hook(self.horizon_lookback_head(hor_check))
        balance = self.balance_check_hook(self.balance_check(hor_lookback, ele_check))
        return balance.float().to(self.device)

def get_LL_parens_model_and_correspondence( n_ctx: int = 20  
) -> Tuple[HookedTransformer, Correspondence, list]:
    """
    For the parenthesis task, returns:
    1. An initialized HookedTransformer for training
    2. The hand-specified correspondence between HL model hooks and LL model nodes for IIT
    3. A list of nodes in the LL model that are not used in the correspondence for SIIT
    """
    #Get HookedTransformer
    cfg = HookedTransformerConfig(
        n_layers = 3,
        d_model = 16,
        n_ctx = n_ctx,
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
        'left_paren_hook':       [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
        'right_paren_hook':      [('blocks.0.attn.hook_z',    Ix[[None, None, 1, None]], None)],
        'elevation_hook' :       [('blocks.0.mlp.hook_post',  Ix[[None]],                None)],
        'elevation_check_hook' : [('blocks.1.mlp.hook_post',  Ix[[None]],                ele_neurons)],
        'horizon_check_hook' :   [('blocks.1.mlp.hook_post',  Ix[[None]],                hor_neurons)], 
        'horizon_lookback_hook': [('blocks.2.attn.hook_z',    Ix[[None, None, 1, None]], None)],
        'balance_check_hook' :   [('blocks.2.mlp.hook_post',    Ix[[None]],                None)]
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
        ('blocks.1.attn.hook_z', [0]), 
        ('blocks.1.attn.hook_z', [1]), 
        ('blocks.2.attn.hook_z', [0]),
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


def test_HL_parens_components():
    """ 
    Checks performance on a few pre-defined sequence inputs:
        1. [BOS] ( ) ( ) ( ) [PAD] [PAD] [PAD] [PAD]
        2. [BOS] ( ( ( ( ( ) ) ) [PAD] [PAD]
        3. [BOS] ) ( ) ( ) ( ) ( ) (
        4. [BOS] ( ) ( ) ( ) ( ) ( )
    """
    
    tokens = [
        [3, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [3, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        [3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    ]
    true_lefts = [
        [ 0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5]
    ]
    true_rights = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5]
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
    true_hor_lookback = [
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True]
    ]
    true_balance_check = [
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True]
    ]
    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_elevations = t.Tensor(true_elevations).to(int)
    true_ele_check = t.Tensor(true_ele_check).to(bool)
    true_hor_check = t.Tensor(true_hor_check).to(bool)
    true_hor_lookback = t.Tensor(true_hor_lookback).to(bool)
    true_balance_check = t.Tensor(true_balance_check).to(bool)

    balance_checker = HighLevelParensBalanceChecker()
    balanced, cache = balance_checker.run_with_cache(tokens)
    assert t.allclose(cache['left_paren_hook'], true_lefts)
    assert t.allclose(cache['right_paren_hook'], true_rights)
    assert t.allclose(cache['elevation_hook'], true_elevations)
    assert t.allclose(cache['elevation_check_hook'], true_ele_check)
    assert t.allclose(cache['horizon_check_hook'], true_hor_check)
    assert t.allclose(cache['horizon_lookback_hook'], true_hor_lookback)
    assert t.allclose(cache['balance_check_hook'], true_balance_check)
    assert t.allclose(balanced, true_balance_check.float())
    print("All tests passed!")

class ParensDatasetBase(ABC):
    
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: Optional[int] = None
    ):
        self.N_samples = N_samples
        self.n_ctx = n_ctx
        self.map_dict = {
            0: ' (',
            1: ' )',
            2: ' PAD',
            3: 'BOS'
        }

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
        
    def passes_balance(self,  sample):
        return (sample == 0).sum() == (sample == 1).sum()
        
    def passes_elevation(self, sample):
        mod = np.copy(sample)
        mod[mod == 1] = -1
        mod[mod == 0] = 1
        mod[mod == 2] = 0
        mod[mod == 3] = 0
        ele = np.cumsum(mod)
        return ele.min() >= 0
    
    def passes_test(self, sample):
        return self.passes_balance(sample)*self.passes_elevation(sample)

    @abstractmethod
    def generate_tokens(self):
        pass

    def map_tokens_to_str(self):
        # Vectorized mapping using numpy
        vectorized_map = np.vectorize(self.map_dict.get)
        self.str_tokens =  vectorized_map(self.tokens)

    def generate_labels(self):
        new_markers = np.zeros(self.tokens.shape[0], dtype=int)
        for i,sample in enumerate(self.tokens):
            if self.passes_test(sample):
                new_markers[i] = 1
            elif (not self.passes_elevation(sample)) and self.passes_balance(sample):
                new_markers[i] = 2
            elif (not self.passes_balance(sample)) and self.passes_elevation(sample):
                new_markers[i] = 3
        self.labels = np.copy(new_markers)
        self.labels[self.labels != 1] = 0
        self.markers = new_markers

    def get_dataset(self):
        return self.dataset

class BalancedParensDataset(ParensDatasetBase):
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: int = 40,
        seed: int = 42
    ):
        np.random.seed(seed)
        super().__init__(N_samples, n_ctx)

    def generate_tokens(self):
        # generate a bunch of tokens of the correct context length
        #empirically if we want ~20_000 balanced samples we need ~25_000 initial samples.
        # (assuming context is long enough)
        N = int(self.N_samples * 5 / 4) 
        first = np.random.binomial(1, 0.25, (N, 1))
        rest = np.random.binomial(1, 0.5, (N, self.n_ctx - 3)) #account for bos, pad, and first paren
        samples = np.concatenate([first, rest], axis=1)
            
        #Look for balanced subsets of each of the generated samples and store those.
        new_strings = []
        pads = np.ones_like(samples[0,:])*2
        sample_progress_bar = tqdm(samples, desc=f"Finding sample substrings", leave=False, position=0)
        for sample in sample_progress_bar:
            for i in range(2, sample.size):
                if i % 2 == 1: 
                    continue
                if self.passes_test(sample[:i])\
                or ((not self.passes_elevation(sample[:i])) and self.passes_balance(sample[:i]))\
                or ((not self.passes_balance(sample[:i])) and self.passes_elevation(sample[:i])):
                    new = np.copy(pads)
                    new[:i] = sample[:i]
                    new_strings.append(new)
        sample_progress_bar.close()
        all_samples = np.concatenate((samples, np.stack(new_strings)), axis=0)
        all_samples = np.unique(all_samples, axis=0) #just keep unique samples

        #Label each sample:
        # 0 - passes no test
        # 1 - passes all test
        # 2 - passes balance; fails elevation
        # 3 - passes elevation; fails balance
        markers = np.zeros(all_samples.shape[0])
        for i,sample in enumerate(all_samples):
            if self.passes_test(sample):
                markers[i] = 1
            elif (not self.passes_elevation(sample)) and self.passes_balance(sample):
                markers[i] = 2
            elif (not self.passes_balance(sample)) and self.passes_elevation(sample):
                markers[i] = 3

        # Figure out which label has the fewest number of samples; dataset will be 4x that many.
        unique_labels, counts = np.unique(markers, return_counts=True)
        min_count = np.min(counts)

        #make sure we have all 4 types
        if len(unique_labels) < 4:
            
            new_samples = []
            if 2 not in unique_labels:
                pass_indices  = np.array(np.where(markers == 1)[0], dtype=int)
                #make min_count copies of items from label 1 but ruin the elevation while preserving balance.
                for i in range(min_count):
                    idx = np.random.choice(len(pass_indices))
                    this_sample = all_samples[pass_indices[idx]]
                    pass_indices = np.delete(pass_indices, idx)
                    # swap a ( and a )
                    while self.passes_elevation(this_sample):
                        zero_spots = np.array(np.where(this_sample == 0)[0], dtype=int)
                        zero_idx = zero_spots[np.random.choice(len(zero_spots))]
                        one_spots = np.array(np.where(this_sample == 1)[0], dtype=int)
                        one_idx = one_spots[np.random.choice(len(one_spots))]
                        this_sample[zero_idx] = 1
                        this_sample[one_idx] = 0
                    new_samples.append(this_sample)
            if 3 not in unique_labels:
                pass_indices  = np.array(np.where(markers == 1)[0], dtype=int)
                #make min_count copies of items from label 1 but ruin the balance while keeping elevation.
                for i in range(min_count):
                    idx = np.random.choice(len(pass_indices))
                    this_sample = all_samples[pass_indices[idx]]
                    pass_indices = np.delete(pass_indices, idx)
                    # turn a ) into a (
                    zero_spots = np.array(np.where(this_sample == 1)[0], dtype=int)
                    idx = np.random.choice(len(zero_spots))
                    this_sample[idx] = 0
                    new_samples.append(this_sample)
            #update samples and markers
            all_samples = np.concatenate((all_samples, np.stack(new_samples)), axis=0)
            all_samples = np.unique(all_samples, axis=0) #just keep unique samples
    
            markers = np.zeros(all_samples.shape[0])
            for i,sample in enumerate(all_samples):
                if self.passes_test(sample):
                    markers[i] = 1
                elif (not self.passes_elevation(sample)) and self.passes_balance(sample):
                    markers[i] = 2
                elif (not self.passes_balance(sample)) and self.passes_elevation(sample):
                    markers[i] = 3
    
            # Figure out which label has the fewest number of samples; dataset will be 4x that many.
            unique_labels, counts = np.unique(markers, return_counts=True)
            min_count = np.min(counts)

        
        # Initialize a list to store the sampled indices
        sampled_indices = []
        
        # For each label, shuffle the indices and select the first min_count elements
        for label in unique_labels:
            indices = np.where(markers == label)[0]
            np.random.shuffle(indices)
            sampled_indices.append(indices[:min_count])

        samples = np.concatenate([all_samples[idx] for idx in sampled_indices], axis=0)
        
        #verification that dataset is balanced
        new_markers = np.zeros(samples.shape[0])
        for i,sample in enumerate(samples):
            if self.passes_test(sample):
                new_markers[i] = 1
            elif (not self.passes_elevation(sample)) and self.passes_balance(sample):
                new_markers[i] = 2
            elif (not self.passes_balance(sample)) and self.passes_elevation(sample):
                new_markers[i] = 3
                
        new_unique_labels, new_counts = np.unique(new_markers, return_counts=True)
        for count in new_counts:
            assert count == min_count

        #add BOS token
        self.tokens = np.concatenate([3*np.ones((samples.shape[0], 1)), samples, 2*np.ones((samples.shape[0], 1))], axis=1).astype(int)

class SequentialParensDataset(ParensDatasetBase):
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: Optional[int] = None
    ):
        # check if n_ctx is long enough for N_samples
        max_n_ctx = self.num_places(N_samples-1) + 1
        if n_ctx is not None and n_ctx < max_n_ctx:
            raise ValueError(f"specific n_ctx is too short for N_samples! Need n_ctx >= {max_n_ctx}.")
        super().__init__(N_samples, n_ctx if n_ctx is not None else max_n_ctx)

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
        self.tokens = []
        for i in range(self.N_samples):
            parentheses = self.sample_parentheses(i)
                
            # Using map with a lambda function
            binary_mapping = list(map(lambda x: 0 if x == '(' else 1, parentheses))
            n_pad = self.n_ctx - (len(binary_mapping) + 1)
            self.tokens.append([3] + binary_mapping + n_pad*[2,]) 
        self.tokens = np.array(tokens).astype(int)


    
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
        'pad_token': 'PAD',
        'mask_token': '[MASK]'
    })
    return hf_tokenizer

def create_paren_checker_tokenizer() -> PreTrainedTokenizerFast:
    # create tokenizer
    # Define your simple vocabulary
    vocab = {'BOS': 3, '(': 0, ')': 1, 'PAD': 2, 'UNK' : 4}
    hf_tokenizer = create_tokenizer(vocab)
    
    # Test the tokenizer
    encoded = hf_tokenizer.encode("BOS ( ) ( ) PAD PAD PAD")
    decoded = hf_tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    return hf_tokenizer