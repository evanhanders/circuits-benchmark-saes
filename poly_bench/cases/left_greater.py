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

class GreaterThan(t.nn.Module):
    """ Calculates if there are more left parens than right parens """
    
    def forward(self, lefts: Int[t.Tensor, "batch seq"], rights: Int[t.Tensor, "batch seq"]) -> Bool[t.Tensor, "batch seq"]:
        return lefts > rights


class HighLevelLeftGreater(HookedRootModule):

    def __init__(self, device='cpu', d_vocab=4):
        super().__init__()
        self.d_vocab = d_vocab
        
        self.input_hook = HookPoint()
        self.left_parens = TokenCountHead(PAREN_REVERSE_VOCAB['('])
        self.right_parens =TokenCountHead(PAREN_REVERSE_VOCAB[')'])
        self.paren_counts_hook = HookPoint()

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
            'paren_counts_hook' :    [('blocks.0.attn.hook_z',    Ix[[None, None, 0, None]], None)],
            'mlp0_hook':             [('blocks.0.mlp.hook_post',  Ix[[None]], None)],
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

        left_parens = self.left_parens(tokens)
        right_parens = self.right_parens(tokens)
        parens = self.paren_counts_hook(t.stack([left_parens, right_parens]))
        left_parens = parens[0]
        right_parens = parens[1]

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
    true_parens = t.stack([true_lefts, true_rights])
    true_mlp0_check = t.Tensor(true_mlp0_check).to(bool)
    true_output = t.nn.functional.one_hot(t.Tensor(true_output).to(int), num_classes=4).float()

    checker = HighLevelLeftGreater()
    output, cache   = checker.run_with_cache((tokens, None, None))
    assert t.allclose(cache['paren_counts_hook'], true_parens)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(output, true_output)
    print("All left greater tests passed!")

    return True


class LeftGreaterDataset:
    
    def __init__(
        self, 
        N_samples: int, 
        n_ctx: Optional[int] = None,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.N_samples = N_samples
        self.n_ctx = n_ctx
        self.map_dict = PAREN_VOCAB

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
        
    def left_greater(self,  sample):
        return np.cumsum(sample == 0) > np.cumsum(sample == 1)
    
    

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
        
    def _generate_left_greater(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, left_greater=True)
    
    def _generate_left_not_greater(self, N_samples, n_ctx):
        return self._generate_token_subset(N_samples, n_ctx, left_greater=False)

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
            PAREN_REVERSE_VOCAB['BOS']*np.ones((dataset.shape[0], 1)), #BOS
            dataset, 
            PAREN_REVERSE_VOCAB[' PAD']*np.ones((dataset.shape[0], 1)) #pad
        ], axis=1).astype(int)

    def map_tokens_to_str(self):
        # Vectorized mapping using numpy
        vectorized_map = np.vectorize(self.map_dict.get)
        self.str_tokens = vectorized_map(self.tokens)

    def generate_labels(self):
        new_markers = np.zeros(self.tokens.shape, dtype=int)
        for i,sample in enumerate(self.tokens):
            sample = sample[1:]
            new_markers[i,1:][self.left_greater(sample)] = 1
        self.markers = new_markers
        self.labels = np.copy(self.markers)
        self.labels[:,0] = 2 #set pad as answer for bos token.
        self.labels = t.nn.functional.one_hot(t.tensor(self.labels), num_classes=4).float().numpy()

    def get_dataset(self):
        return self.dataset
    
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
