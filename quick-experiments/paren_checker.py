from abc import ABC, abstractmethod 
from jaxtyping import Float, Int, Bool
from typing import Optional, Callable, Tuple

import torch as t
from torch.utils.data import ConcatDataset
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
        tok_clone[:,1] = 0 #don't count job id token

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
        tok_clone[:,1] = 0 #don't count job id token

        #we'll count left to right.
        rights = t.cumsum(tok_clone, dim=1).to(int)

        return rights

class ElevationCalculator(t.nn.Module):
    """ Calculates the elevation at each position in the context"""
    
    def forward(self, lefts: Int[t.Tensor, "batch seq"], rights: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        return lefts - rights

class GreaterThan(t.nn.Module):
    """ Calculates if a number is even or not  """

    def forward(self, left: Int[t.Tensor, "batch seq"], right: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq"]:
        return (left > right)

class TaskIdentifier(t.nn.Module):

    def forward(self, tokens: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch"]:
        jobs = t.zeros(tokens.shape[0]).to(int)
        jobs[tokens[:,1] == 1] = 1 #if token in position 1 is a 1, do job 2. Else, job 1.
        return jobs

class CheckElevation(t.nn.Module):
    """ Checks if the elevation in token position -1 is 0.
        Returns 1 if so and 0 if false.
    """

    def forward(self, elevations: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch"]:
        elevation_bool = t.ones(elevations.shape[0], dtype=t.bool)
        elevation_bool[elevations[:,-1].nonzero()] = 0
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
    ) -> Bool[t.Tensor, "batch"]:
        #this basically just gives True if horizon has never been violated before or now; false if horizon has been violated
        return t.cumprod(horizon_check, dim=1)[:,-1].bool() 

class OutputChecks(t.nn.Module):

    def forward(
        self,
        task_id : Int[t.Tensor, "batch"],
        left_greater : Int[t.Tensor, "batch seq"],
        ele_check : Bool[t.Tensor, "batch"],
        bal_check : Bool[t.Tensor, "batch"]
    ) -> Int[t.Tensor, "batch"]:
        if task_id.min() < 0 or task_id.max() > 1:
            raise ValueError("Task ID must be 0 or 1")
        result = t.zeros(task_id.shape[0]).to(int)
        result[task_id == 0] = (ele_check*bal_check).to(int)[task_id == 0]
        result[task_id == 1] = (left_greater[:,-1] == 1).to(int)[task_id == 1]
        return result

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
        self.left_parens = LeftParenCountHead()
        self.left_parens_hook = HookPoint()
        self.right_parens = RightParenCountHead()
        self.right_parens_hook = HookPoint()
        self.task_hook = HookPoint()
        self.task_id = TaskIdentifier()

        self.greater_than = GreaterThan()
        self.elevation_calc = ElevationCalculator()
        self.greater_hook = HookPoint()
        self.elevation_hook = HookPoint()
        self.mlp0_hook = HookPoint()

        self.elevation_checker = CheckElevation()
        self.horizon_checker = CheckHorizon()
        self.mlp1_hook = HookPoint()

        self.horizon_lookback_head = HorizonLookbackHead()
        self.horizon_lookback_hook = HookPoint()
        self.output_check = OutputChecks()
        self.output_check_hook = HookPoint()
        self.device = device
        self.setup()
    
    def is_categorical(self) -> bool:
        return True

    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Tensor, t.Tensor]) -> Float[t.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)
        left_parens = self.left_parens_hook(self.left_parens(tokens))
        right_parens = self.right_parens_hook(self.right_parens(tokens))
        task_id = self.task_hook(self.task_id(tokens))

        elevation = self.elevation_hook(self.elevation_calc(left_parens, right_parens))
        left_greater = self.greater_hook(self.greater_than(left_parens, right_parens))
        mlp0 = self.mlp0_hook(t.stack((elevation, left_greater)))
        elevation = mlp0[0]
        left_greater = mlp0[1]
        
        ele_check = self.elevation_checker(elevation)
        hor_check = self.horizon_checker(elevation)
        hook_mlp1 = self.mlp1_hook(t.cat((ele_check.unsqueeze(1), hor_check), dim=1))
        ele_check = hook_mlp1[:,0]
        hor_check = hook_mlp1[:,1:hor_check.shape[1]+1]
        hor_lookback = self.horizon_lookback_hook(self.horizon_lookback_head(hor_check))


    
        output = self.output_check_hook(self.output_check(task_id, left_greater, hor_lookback, ele_check))
        true_output = t.zeros((output.shape[0], 1, 2)).to(self.device)
        true_output[output == 0, :, 0] = 1
        true_output[output == 1, :, 1] = 1
        
        return true_output.float().to(self.device)

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
        d_model = 32,
        n_ctx = n_ctx,
        d_head = 8,
        d_vocab = 4,
        act_fn = "relu",
    )
    model = HookedTransformer(cfg)

    #Get Correspondence
    corr = {
        'input_hook' :           [('hook_embed', Ix[[None]],                None)],
        'left_parens_hook' :     [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
        'right_parens_hook' :    [('blocks.0.attn.hook_z',    Ix[[None, None, 1, None]], None)],
        'task_hook':             [('blocks.0.attn.hook_z',    Ix[[None, None, 2, None]], None)],
        'mlp0_hook':             [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
        'mlp1_hook' :            [('blocks.1.mlp.hook_post',  Ix[[None]], None)],
        'horizon_lookback_hook': [('blocks.2.attn.hook_z',    Ix[[None, None, 3, None]], None)],
        'output_check_hook' :    [('blocks.2.mlp.hook_post',  Ix[[None]], None)]
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
        ('blocks.0.attn.hook_z', [3]),
        ('blocks.1.attn.hook_z', [0, 1, 2, 3]), 
        ('blocks.2.attn.hook_z', [0, 1, 2])
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


def test_HL_parens_balancer_components():
    # parens balance check
    tokens = [
        [3, 0, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        [3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [3, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    ]
    true_lefts = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3],
    ]
    true_rights = [
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7],
    ]
    true_mlp0_check = [ #elevations
        [ 0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  1,  2,  3,  4,  5,  4,  3,  2,  2,  2],
        [ 0,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0],
        [ 0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [ 0,  0, -1, -2, -1, -2, -3, -2, -3, -4, -3, -4],
    ]
    true_mlp1_check = [ #first element is ele; rest are horizon
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True, False,  True, False,  True, False,  True, False,  True, False, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [False,  True,  True, False, False, False, False, False, False, False, False, False, False],
    ]
    true_hor_lookback = [ True,  True, False, True, False,]
    true_task_id = [ 0, 0, 0, 0, 0,]
    true_output  = [ 1, 0, 0, 1, 0, ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_mlp0_check = t.Tensor(true_mlp0_check).to(int)
    true_mlp0_check = t.stack((true_mlp0_check, (true_lefts > true_rights).to(int)))
    true_mlp1_check = t.Tensor(true_mlp1_check).to(bool)
    true_hor_lookback = t.Tensor(true_hor_lookback).to(bool)
    true_task_id = t.Tensor(true_task_id).to(int)
    true_output = t.Tensor(true_output).to(int)
    

    balance_checker = HighLevelParensBalanceChecker()
    output, cache   = balance_checker.run_with_cache((tokens, None, None))
    # print(cache['right_parens_hook'] - true_rights)
    assert t.allclose(cache['left_parens_hook'], true_lefts)
    assert t.allclose(cache['right_parens_hook'], true_rights)
    assert t.allclose(cache['task_hook'], true_task_id)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(cache['mlp1_hook'], true_mlp1_check)
    assert t.allclose(cache['horizon_lookback_hook'], true_hor_lookback)
    assert t.allclose(cache['output_check_hook'], true_output)
    print("All tests passed!")
    return True

def test_HL_parens_gtr_components():
    # parens balance check
    tokens = [
        [3, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [3, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        [3, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [3, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    ]
    true_lefts = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3],
    ]
    true_rights = [
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7],
    ]
    true_task_id = [ 1, 1, 1, 1, 1,]
    true_output  = [ 0, 1, 0, 0, 0, ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    elevation = true_lefts - true_rights
    true_mlp0_check = t.stack((elevation, true_lefts > true_rights)).to(int)
    ele_check = elevation[:,-1] == 0
    hor_check = elevation >= 0
    true_mlp1_check = t.cat([ele_check[:,None], hor_check], dim=1)
    true_task_id = t.Tensor(true_task_id).to(int)
    true_output = t.Tensor(true_output).to(int)
    

    balance_checker = HighLevelParensBalanceChecker()
    output, cache   = balance_checker.run_with_cache((tokens, None, None))
    # print(cache['right_parens_hook'] - true_rights)
    assert t.allclose(cache['left_parens_hook'], true_lefts)
    assert t.allclose(cache['right_parens_hook'], true_rights)
    assert t.allclose(cache['task_hook'], true_task_id)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(cache['mlp1_hook'], true_mlp1_check)
    assert t.allclose(cache['output_check_hook'], true_output)
    print("All tests passed!")
    return True

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
    
    def passes_balance_test(self, sample):
        return self.passes_balance(sample)*self.passes_elevation(sample)

    def passes_left_gtr_right(self, sample):
        return (sample == 0).sum() > (sample == 1).sum()
    
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
            if sample[1] == 0:
                sample = sample[2:]
                if self.passes_balance_test(sample):
                    new_markers[i] = 1
                elif (not self.passes_balance(sample)) and self.passes_elevation(sample):
                    new_markers[i] = 2
                elif (not self.passes_elevation(sample)) and self.passes_balance(sample):
                    new_markers[i] = 3
            elif sample[1] == 1:
                sample = sample[2:]
                if self.passes_left_gtr_right(sample):
                    new_markers[i] = 1
            else:
                raise NotImplementedError("Task 1+ logic not implemented for marking")
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels[self.labels != 1] = 0

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

    def _generate_token_subset(self, N_samples, n_ctx, passes_balance=True, passes_elevation=True):
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
            if passes_elevation:
                mask = min_values >= 0
            else:
                mask = min_values < 0
            masked_samples = shuffled[mask,:]
            good_samples.append(masked_samples[:remaining_samples])
            remaining_samples -= good_samples[-1].shape[0]
            
        return t.unique(t.cat(good_samples, dim=0), dim=0)
        
    def _generate_balanced_tokens(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=True, passes_elevation=True)

    def _generate_elevation_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=True, passes_elevation=False)
        
    def _generate_balance_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=False, passes_elevation=True)
        
    def _generate_absolute_failures(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, passes_balance=False, passes_elevation=False)
        
    def generate_tokens(self):

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        balanced = self._generate_balanced_tokens(self.N_samples, self.n_ctx - 3)
        fail_ele = self._generate_elevation_failures(self.N_samples // 2, self.n_ctx - 3)
        fail_bal = self._generate_balance_failures(self.N_samples // 2, self.n_ctx - 3)
        fail_both = self._generate_absolute_failures(self.N_samples // 2, self.n_ctx - 3)

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
            3*np.ones((dataset.shape[0], 1)), #BOS
            0*np.ones((dataset.shape[0], 1)), #task id
            dataset, 
            2*np.ones((dataset.shape[0], 1)) #pad
        ], axis=1).astype(int)



class LeftGreaterParensDataset(ParensDatasetBase):
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: int = 40,
        seed: int = 42
    ):
        np.random.seed(seed)
        super().__init__(N_samples, n_ctx)

    def _generate_token_subset(self, N_samples, n_ctx, left_greater=True):
        """ Samples that fail both tests """
        assert n_ctx % 2 == 0, "n_ctx must be even."
        
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
            samples = t.stack(samples)
            
            # shuffle
            indices = t.stack([t.randperm(n_ctx) for _ in range(generated_samples)])
            shuffled = t.gather(samples, 1, indices)
            good_samples.append(shuffled)
            remaining_samples -= good_samples[-1].shape[0]
            
        return t.unique(t.cat(good_samples, dim=0), dim=0)
    
    def generate_tokens(self):

        #Generate a bunch of examples -- we'll only use a fraction but due to uniqueness we won't get as many as we want.
        greater = self._generate_token_subset(self.N_samples, self.n_ctx - 3, left_greater=True)
        less = self._generate_token_subset(self.N_samples, self.n_ctx - 3, left_greater=False)

        dataset = t.cat([
            greater[:self.N_samples // 2],
            less[:self.N_samples // 2],
        ], dim = 0)
        dataset[dataset == 1]  = 0 #(
        dataset[dataset == -1] = 1 #)
        dataset = dataset[t.randperm(dataset.shape[0]),:] #shuffle the dataset.

        #add BOS token to beginning and pad to end
        self.tokens = np.concatenate([
            3*np.ones((dataset.shape[0], 1)), #BOS
            1*np.ones((dataset.shape[0], 1)), #task id
            dataset, 
            2*np.ones((dataset.shape[0], 1)) #pad
        ], axis=1).astype(int)

class TwoTaskParensDataset():

    def __init__(self,
                 N_samples: int,
                 n_ctx: Optional[int] = None,
                 seed: int = 42,
                 dataset_types: list[ParensDatasetBase] = [BalancedParensDataset, LeftGreaterParensDataset]
                 ):
        self.datasets = [dataset_type(N_samples=N_samples // 2, n_ctx = n_ctx, seed=seed) for dataset_type in dataset_types]
        self.n_ctx = n_ctx
        self.N_samples = N_samples
        self.map_dict = self.datasets[0].map_dict
        self.tokens = np.concatenate([dataset.tokens for dataset in self.datasets])
        self.str_tokens = np.concatenate([dataset.str_tokens for dataset in self.datasets])
        self.labels = np.concatenate([dataset.labels for dataset in self.datasets])
        self.markers = np.concatenate([dataset.markers for dataset in self.datasets])

        self.dataset = Dataset.from_dict({
                'tokens' : self.tokens,
                'str_tokens' : self.str_tokens,
                'labels': self.labels,
                'markers' : self.markers
            })
    

    def get_dataset(self):
        return self.dataset



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
        self.tokens = np.array(tself.okens).astype(int)


    
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