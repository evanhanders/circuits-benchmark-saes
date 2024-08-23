from collections import defaultdict
from typing import Optional, List
from jaxtyping import Int, Float

import torch as t
from torch import nn, Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint # type: ignore
from transformer_lens import HookedTransformerConfig, HookedTransformer # type: ignore
from iit.utils.index import TorchIndex, Ix # type: ignore
from iit.utils.correspondence import Correspondence # type: ignore
from iit.utils.iit_dataset import train_test_split, IITDataset # type: ignore
from iit.utils.nodes import HLNode #type: ignore

from .cases.poly_case import PolyCase, PolyBenchDataset # type: ignore
from .utils import SimpleDataset # type: ignore


class PolyModelDataset:

    def __init__(self, datasets: list[PolyBenchDataset], n_ctx: int, train_frac=0.8, seed=42):
        N_samples = None
        for dataset in datasets:
            if dataset.n_ctx >= n_ctx:
                raise ValueError("All datasets must have n_ctx less than the combined model's n_ctx.")
            if N_samples is None:
                N_samples = dataset.N_samples
            elif N_samples != dataset.N_samples:
                raise ValueError("All datasets must have the same number of samples.")
            
        self.datasets = datasets
        self.n_ctx = n_ctx

        # reset the dataset.tokens attribute by adding a task_id token to the beginning of the sequence and padding to the end.
        max_d_vocab = 0
        for i, dataset in enumerate(datasets):
            padding = [dataset.map_dict['PAD']]*(n_ctx - dataset.n_ctx - 1) 
            dataset.tokens = t.tensor([ [i] + list(seq) + padding for seq in dataset.tokens])
            dataset.map_tokens_to_str()
            dataset.generate_labels(skip_first=True)
            if dataset.labels.shape[-1] > max_d_vocab:
                max_d_vocab = dataset.labels.shape[-1]
        
        # ensure that all labels have the same d_vocab
        for i, dataset in enumerate(datasets):
            #ensure that last dim of dataset.labels has length max_d_vocab; if not, expand with zeros.
            if dataset.labels.shape[-1] < max_d_vocab:
                label_padding = t.zeros((dataset.labels.shape[0], dataset.labels.shape[1], max_d_vocab - dataset.labels.shape[-1]))
                dataset.labels = t.cat([t.tensor(dataset.labels), label_padding], dim=-1)
            else:
                dataset.labels = t.tensor(dataset.labels)
            dataset.build_dataset()


        #combine all datasets into one.
        tokens = t.cat([d.tokens if isinstance(d.tokens, t.Tensor) else t.tensor(d.tokens) for d in datasets], dim=0)
        labels = t.cat([d.labels if isinstance(d.labels, t.Tensor) else t.tensor(d.labels) for d in datasets], dim=0)
        markers = t.cat([d.markers if isinstance(d.markers, t.Tensor) else t.tensor(d.markers) for d in datasets], dim=0)

        #shuffle the dataset
        idx = t.randperm(tokens.shape[0])
        tokens = tokens[idx]
        labels = labels[idx]
        markers = markers[idx]

        #Build IIT datasets
        self.dataset = SimpleDataset(
            inputs = tokens,
            targets = labels,
            markers = markers
        )
        train_dataset, test_dataset = train_test_split(
            self.dataset, test_size=1-train_frac, random_state=42
        )
        self.train_set = IITDataset(train_dataset, train_dataset, seed=seed)
        self.test_set = IITDataset(test_dataset, test_dataset, seed=seed)
    
    def get_IIT_train_test_set(self):
        return self.train_set, self.test_set

    def get_dataset(self):
        return self.dataset        
        

class PolyHLModel(HookedRootModule):

    def __init__(
            self, 
            hl_classes: list[PolyCase], 
            attn_suffix: str = 'attn.hook_z', #force all attn hooks to have the same suffix
            mlp_suffix: str = 'mlp.hook_post', #force all mlp hooks to have the same suffix'
            size_expansion: int = 1,
            ):
        super().__init__()
        self.attn_suffix = attn_suffix
        self.mlp_suffix = mlp_suffix
        self.hl_classes = hl_classes
        self.hl_models = [hl_class() for hl_class in hl_classes]
        self.corrs = [model.get_correspondence() for model in self.hl_models]
        self.cfgs = [model.get_ll_model_cfg() for model in self.hl_models]

        self.found_space = False

        # # TODO: For circuits-benchmark integration
        # self.tracr_d_heads = [mod.W_Q.shape[-1] for mod in self.hl_models]
        # self.tracr_d_mlps = [mod.W_in[0].shape[1] for mod in self.hl_models]
        # self.tracr_d_models = [mod.cfg.d_model for mod in self.hl_models]
        # self.tracr_n_ctx = [mod.cfg.n_ctx for mod in self.hl_models]

        self.d_model_list = [cfg.d_model for cfg in self.cfgs]
        self.n_layers_list = [cfg.n_layers for cfg in self.cfgs]
        self.n_heads_list = [cfg.n_heads for cfg in self.cfgs]
        self.n_ctx_list = [cfg.n_ctx for cfg in self.cfgs]
        self.d_head_list = [cfg.d_head for cfg in self.cfgs]
        self.d_mlp_list = [cfg.d_mlp for cfg in self.cfgs]
        self.d_vocab_list = [cfg.d_vocab for cfg in self.cfgs]

        self.cfg = HookedTransformerConfig(
            n_layers = max(self.n_layers_list),
            d_model = size_expansion*max(self.d_model_list),
            n_ctx = max(self.n_ctx_list) + 1,
            d_head = size_expansion*max(self.d_head_list),
            d_vocab = max(self.d_vocab_list),
            act_fn = "relu"
        )
        self.device = self.cfg.device

        # We need to store the shapes of the attn results because hook_result and hook_z have different shapes.
        self.attn_shapes = [] 
        
        for model_number, corr in enumerate(self.corrs):
            if corr.suffixes['attn'] == 'attn.hook_result':
                self.attn_shapes.append(self.d_model_list[model_number])
            else:
                self.attn_shapes.append(self.d_head_list[model_number])
            
        self.attn_shape = max(self.attn_shapes)

        #make hooks for each attn head and each mlp; we'll only use a subset of them.
        self.input_hook = HookPoint()
        self.task_hook = HookPoint()
        self.attn_hooks = nn.ModuleList([nn.ModuleList([HookPoint() for _ in range(self.cfg.n_heads)]) for _ in range(self.cfg.n_layers)])
        self.mlp_hooks = nn.ModuleList([HookPoint() for _ in range(self.cfg.n_layers)])

        # Loop through each layer, (head and mlp) in the combined model.
        # Add a list of hooks that go to that head/mpl in the LL models.
        self.corr_mapping: dict[str, List[Optional[List[HLNode]]]] = defaultdict(list)
        corr_dict: dict[str, List[Optional[tuple[str, TorchIndex, Optional[t.Any]]]]] = defaultdict(list)

        task_id_set = False
        corr_dict['input_hook'] = [('blocks.0.hook_resid_pre', Ix[[None]], None),]
        for layer in range(self.cfg.n_layers):
            # create MLP hooks
            mlp_hook_name = f'blocks.{layer}.{self.mlp_suffix}'
            for i, corr in enumerate(self.corrs):
                not_yet_used = True
                for k, v in corr.items():
                    for node in v:
                        if node.name == f"blocks.{layer}.{corr.suffixes['mlp']}":
                            if not_yet_used:
                                self.corr_mapping[mlp_hook_name].append([k])
                                not_yet_used = False
                            else:
                                assert isinstance(self.corr_mapping[mlp_hook_name][i], list)
                                self.corr_mapping[mlp_hook_name][i].append([k]) #type: ignore
                if not_yet_used:
                    self.corr_mapping[mlp_hook_name].append(None)
                else:
                    corr_dict[f'mlp_hooks.{layer}'] = [(mlp_hook_name, Ix[[None]], None),]
            
            # create Attn hooks
            attn_hook_name = f'blocks.{layer}.{self.attn_suffix}'
            for head in range(self.cfg.n_heads):
                corr_key = f'{attn_hook_name}.{head}'
                self.corr_mapping[corr_key] = []
                for i, corr in enumerate(self.corrs):
                    not_yet_used = True
                    for k, v in corr.items():
                        for node in v:
                            if node.name == f"blocks.{layer}.{corr.suffixes['attn']}" and node.index.as_index[2] == head:
                                if not_yet_used:
                                    self.corr_mapping[corr_key].append([k])
                                    not_yet_used = False
                                else:
                                    assert isinstance(self.corr_mapping[corr_key], list)
                                    self.corr_mapping[corr_key][i].append([k]) #type: ignore
                    if not_yet_used:
                        self.corr_mapping[corr_key].append(None)
                    else:
                        corr_dict[f'attn_hooks.{layer}.{head}'] = [(attn_hook_name, Ix[[None,None,head,None]], None),]  
                if not task_id_set and all([val is None for val in self.corr_mapping[corr_key]]):
                    corr_dict['task_hook'] = [(attn_hook_name, Ix[[None,None,head,None]], None),]
                    task_id_set = True     
            
        self.corr = Correspondence.make_corr_from_dict(corr_dict, suffixes={'attn': self.attn_suffix, 'mlp': self.mlp_suffix})
        self.setup()
    
    def is_categorical(self) -> bool:
        return True

    def get_ll_model(self, seed: Optional[int] = None) -> HookedTransformer:
        if seed is not None:
            self.cfg.seed = seed
        self.cfg.init_mode = "xavier_normal"
        return HookedTransformer(self.cfg)

    def get_correspondence(self) -> Correspondence:
        return self.corr

    def sort_output(self, tokens: Tensor, model_outputs: list[Tensor], task_ids: Tensor) -> Tensor:
        outputs = t.zeros((tokens.shape[0], tokens.shape[1], self.cfg.d_vocab)).to(self.cfg.device)
        self.mask = t.ones_like(outputs).to(t.bool)
        self.mask[:,0] = False
        for i, hl_model in enumerate(self.hl_models):
            outputs[task_ids == i, 1:self.n_ctx_list[i]+1,:self.d_vocab_list[i]] = model_outputs[i][task_ids == i].to(self.cfg.device)
            outputs[task_ids == i, 0, 0] = 1

            #TODO: Make & return (or store?) a mask for evaluating the loss on the output.
            self.mask[task_ids == i, self.n_ctx_list[i]+1:] = False
        
        return outputs
    
    def forward(self, inputs: tuple[Int[t.Tensor, "batch seq"], t.Any, t.Any]) -> Float[t.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)
        task_ids = tokens[:, 0]
        task_ids = self.task_hook(task_ids)

        task_tokens = []
        for i, hl_model in enumerate(self.hl_models):

            n_ctx = self.n_ctx_list[i]
            d_vocab = self.d_vocab_list[i]
            task_tokens.append(t.clone(tokens)[:, 1:n_ctx+1])
            bad_tokens_mask = task_tokens[-1] >= d_vocab
            #randomly sample from 2 to d_vocab-1 to replace out-of-task tokens.
            task_tokens[-1][bad_tokens_mask] = t.randint(2, d_vocab-1, (bad_tokens_mask.sum().item(),)).to(t.int)

        # Step 1 -- get all the activations.
        caches = []
        for i, hl_model in enumerate(self.hl_models):
            # # tracr version
            # if isinstance(hl_model, IITHLModel):
            #     hl_model = hl_model.hl_model

            #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            these_tokens = task_tokens[i]
            _, cache = hl_model.run_with_cache((these_tokens, None, None))
            caches.append(cache)
        
        # for k, i in caches[0].items():
        #     print(k, i.shape)
        
        #ActivationCache isn't writable so we need to copy it.
        caches = [{k: i for k, i in cache.items()} for cache in caches]



        # Walk through all of the activations in the caches and figure out how to map them into a single tensor per HL hook.
        for name, hooks in self.corr_mapping.items():
            if self.mlp_suffix not in name and self.attn_suffix not in name:
                continue
            # print(name, hooks)
            data_list = []
            shapes = []
            slices = []
            numel = 0
            for i in range(len(self.hl_models)):
                # print(name, hooks[i])
                if hooks[i] is not None:
                    for hook in hooks[i]: #type: ignore
                        # print(hook, caches[i][hook].shape)
                        shapes.append(caches[i][hook].shape)
                        data_list.append(caches[i][hook].flatten())
                        this_numel = caches[i][hook].numel()
                        slices.append(slice(numel, numel + this_numel))
                        numel += this_numel
            if numel > 0:
                data = t.cat(data_list)
                if self.mlp_suffix in name:
                    layer = int(name.split('.')[1])
                    hook = self.mlp_hooks[layer]
                elif self.attn_suffix in name:
                    layer = int(name.split('.')[1])
                    head = int(name.split('.')[4])
                    hook = self.attn_hooks[layer][head]
                data = hook(data)
                idx = 0
                for i in range(len(self.hl_models)):
                    if hooks[i] is not None:
                        for j, hook in enumerate(hooks[i]): #type: ignore
                            caches[i][hook] = data[slices[idx]].reshape(shapes[idx])
                            idx += 1
                
      

        # Step 3 -- run with a bunch of hooks on the tracr models using the cache, actually doing interventions, and return intervened output.
        model_outputs = []
        for i, hl_model in enumerate(self.hl_models):
            # if isinstance(hl_model, IITHLModel):
            #     hl_model = hl_model.hl_model
            # #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            # encoder = hl_model.tracr_input_encoder
            # pad_id = encoder.encoding_map[TRACR_PAD]

            these_tokens = task_tokens[i]

            # define hooks.

            def simple_replacement_hook(x, hook):
                x[:] = caches[i][hook.name].to(x.device)

            mlp_replacement_hook = simple_replacement_hook
            
            def attn_replacement_hook(x, hook, index: Optional[TorchIndex] = None):
                x[index.as_index] = caches[i][hook.name][index.as_index]

            hooks = []
            #go through self.corr_dict and link up each hook with corresponding hook(s) in HL tracr models.
            for layer in range(self.cfg.n_layers):

                #MLP
                hook_name = f'blocks.{layer}.{self.mlp_suffix}'
                hl_model_hook_name = self.corr_mapping[hook_name][i]
                #unpack from cache
                if hl_model_hook_name is not None:
                    for node in hl_model_hook_name:
                        hooks.append((node.name, simple_replacement_hook))
                
                # Attn
                hook_name = f'blocks.{layer}.{self.attn_suffix}'
                for head in range(self.n_heads_list[i]):
                    hl_model_hook_name = self.corr_mapping[f'{hook_name}.{head}'][i]
                    if hl_model_hook_name is not None:
                        for node in hl_model_hook_name:
                            hooks.append((node.name, simple_replacement_hook))
                            # hooks.append((node.name, partial(attn_replacement_hook, index=TorchIndex([None,None,head,None])))) # for tracr.

            # print(hooks)
            model_output = hl_model.run_with_hooks((these_tokens, None, None), fwd_hooks=hooks)
            model_outputs.append(model_output)
        
        outputs = self.sort_output(tokens, model_outputs, task_ids)

        return outputs
    
    #TODO: implement masking for loss computation. Would be done in an inherited StrictIITModelPair.