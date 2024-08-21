from collections import defaultdict
from functools import partial
from typing import Optional, Callable
from jaxtyping import Int, Float


import torch
import numpy as np
from torch import nn, Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformerConfig, HookedTransformer
from iit.utils.index import TorchIndex, Ix
from iit.utils.correspondence import Correspondence
from iit.utils.nodes import HLNode
from iit.utils.iit_dataset import train_test_split, IITDataset

from circuits_benchmark.utils.iit.iit_hl_model import IITHLModel
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_PAD

from cases.poly_case import PolyCase, PolyBenchDataset
from cases.utils import CustomDataset


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
            dataset.tokens = torch.tensor([ [i] + seq.tolist() + padding for seq in dataset.tokens])
            dataset.map_tokens_to_str()
            dataset.generate_labels(skip_first=True)
            if dataset.labels.shape[-1] > max_d_vocab:
                max_d_vocab = dataset.labels.shape[-1]
        
        # ensure that all labels have the same d_vocab
        for i, dataset in enumerate(datasets):
            #ensure that last dim of dataset.labels has length max_d_vocab; if not, expand with zeros.
            if dataset.labels.shape[-1] < max_d_vocab:
                padding = torch.zeros((dataset.labels.shape[0], dataset.labels.shape[1], max_d_vocab - dataset.labels.shape[-1]))
                dataset.labels = torch.cat([torch.tensor(dataset.labels), padding], dim=-1)
            dataset.build_dataset()


        #combine all datasets into one.
        tokens = torch.cat([torch.tensor(dataset.tokens) for dataset in datasets], dim=0)
        labels = torch.cat([torch.tensor(dataset.labels) for dataset in datasets], dim=0)
        markers = torch.cat([torch.tensor(dataset.markers) for dataset in datasets], dim=0)

        #shuffle the dataset
        idx = torch.randperm(tokens.shape[0])
        tokens = tokens[idx]
        labels = labels[idx]
        markers = markers[idx]

        #Build IIT datasets
        self.dataset = CustomDataset(
            inputs = tokens,
            targets = np.array(labels),
            markers = np.array(markers)
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
            n_layers = max(self.n_layers_list) + 1, #add one for the input layer
            d_model = size_expansion*max(self.d_model_list),
            n_ctx = max(self.n_ctx_list) + 1,
            d_head = size_expansion*max(self.d_head_list),
            d_vocab = max(self.d_vocab_list),
            act_fn = "relu"
        )

        # We need to store the shapes of the attn results because hook_result and hook_z have different shapes.
        self.attn_shapes = [] 
        corr_dict = defaultdict(list)
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
        self.corr_mapping = defaultdict(list)

        corr_dict = {}

        corr_dict[f'task_hook'] = [(f'blocks.0.{self.attn_suffix}', Ix[[None,None,0,None]], None),]
        corr_dict['input_hook'] = [('blocks.1.hook_resid_pre', Ix[[None]], None),] #input hook of ll models is after an MLP
        for layer in range(self.cfg.n_layers):
            if layer == 0:
                continue

            # create MLP hooks
            mlp_hook_name = f'blocks.{layer}.{self.mlp_suffix}'
            for i, corr in enumerate(self.corrs):
                not_yet_used = True
                for k, v in corr.items():
                    for node in v:
                        if node.name == f"blocks.{layer-1}.{corr.suffixes['mlp']}":
                            if not_yet_used:
                                self.corr_mapping[mlp_hook_name].append([k])
                                not_yet_used = False
                            else:
                                self.corr_mapping[mlp_hook_name][i].append([k])
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
                            if node.name == f"blocks.{layer-1}.{corr.suffixes['attn']}" and node.index.as_index[2] == head:
                                if not_yet_used:
                                    self.corr_mapping[corr_key].append([k])
                                    not_yet_used = False
                                else:
                                    self.corr_mapping[corr_key][i].append([k])
                    if not_yet_used:
                        self.corr_mapping[corr_key].append(None)
                    else:
                        corr_dict[f'attn_hooks.{layer}.{head}'] = [(attn_hook_name, Ix[[None,None,head,None]], None),]  
            
        self.corr = Correspondence.make_corr_from_dict(corr_dict, suffixes={'attn': self.attn_suffix, 'mlp': self.mlp_suffix})
        self.setup()
    
    def is_categorical(self) -> bool:
        return True

    def get_ll_model(self, seed: Optional[int] = None) -> HookedTransformer:
        if seed is not None:
            self.cfg.seed = seed
        return HookedTransformer(self.cfg)

    def get_correspondence(self) -> Correspondence:
        return self.corr
    
    def forward(self, inputs: tuple[Int[torch.Tensor, "batch seq"], torch.Any, torch.Any]) -> Float[torch.Tensor, "batch seq logits"]:
        tokens, _, _ = inputs
        tokens = self.input_hook(tokens)
        task_ids = tokens[:, 0]
        task_ids = self.task_hook(task_ids)



        # Step 1 -- get all the activations.
        caches = []
        for i, hl_model in enumerate(self.hl_models):
            # # tracr version
            # if isinstance(hl_model, IITHLModel):
            #     hl_model = hl_model.hl_model

            #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            # encoder = hl_model.tracr_input_encoder
            # pad_id = encoder.encoding_map[TRACR_PAD]
            # d_vocab = hl_model.cfg.d_vocab
            # n_ctx = hl_model.cfg.n_ctx
            pad_id = hl_model.vocab_dict['PAD']
            d_vocab = self.d_vocab_list[i]
            n_ctx = self.n_ctx_list[i]
            task_tokens = torch.clone(tokens)[:, 1:n_ctx+1]
            task_tokens[task_tokens >= d_vocab] = pad_id
            _, cache = hl_model.run_with_cache((task_tokens, None, None))
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
            data = []
            shapes = []
            slices = []
            numel = 0
            for i in range(len(self.hl_models)):
                # print(name, hooks[i])
                if hooks[i] is not None:
                    for hook in hooks[i]:
                        # print(hook, caches[i][hook].shape)
                        shapes.append(caches[i][hook].shape)
                        data.append(caches[i][hook].flatten())
                        this_numel = caches[i][hook].numel()
                        slices.append(slice(numel, numel + this_numel))
                        numel += this_numel
            if numel > 0:
                data = torch.cat(data)
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
                        for j, hook in enumerate(hooks[i]):
                            caches[i][hook] = data[slices[idx]].reshape(shapes[idx])
                            idx += 1
                
      

        # Step 3 -- run with a bunch of hooks on the tracr models using the cache, actually doing interventions, and return intervened output.
        outputs = torch.zeros((tokens.shape[0], tokens.shape[1], self.cfg.d_vocab)).to(self.cfg.device)
        self.mask = torch.ones_like(outputs).to(bool)
        self.mask[:,0] = False
        for i, hl_model in enumerate(self.hl_models):
            # if isinstance(hl_model, IITHLModel):
            #     hl_model = hl_model.hl_model
            # #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            # encoder = hl_model.tracr_input_encoder
            # pad_id = encoder.encoding_map[TRACR_PAD]

            pad_id = hl_model.vocab_dict['PAD']
            task_tokens = torch.clone(tokens)[:, 1:self.n_ctx_list[i]+1]
            task_tokens[task_tokens >= self.d_vocab_list[i]] = pad_id

            # define hooks.

            def simple_replacement_hook(x, hook):
                x[:] = caches[i][hook.name].to(x.device)

            mlp_replacement_hook = simple_replacement_hook
            
            def attn_replacement_hook(x, hook, index: Optional[TorchIndex] = None):
                x[index.as_index] = caches[i][hook.name][index.as_index]

            hooks = []
            #go through self.corr_dict and link up each hook with corresponding hook(s) in HL tracr models.
            for layer in range(self.cfg.n_layers):
                if layer == 0:
                    continue

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
            model_output = hl_model.run_with_hooks((task_tokens, None, None), fwd_hooks=hooks)
            outputs[task_ids == i, 1:self.n_ctx_list[i]+1,:self.d_vocab_list[i]] = model_output[task_ids == i].to(self.cfg.device)
            outputs[task_ids == i, 0, hl_model.vocab_dict['PAD']] = 1
        
            #TODO: Make & return (or store?) a mask for evaluating the loss on the output.
            self.mask[task_ids == i, self.n_ctx_list[i]+1:] = False

        return outputs
    
    #TODO: implement masking for loss computation.
    
    # def get_IIT_loss_over_batch(
    #     self,
    #     base_input: tuple[Tensor, Tensor, Tensor],
    #     ablation_input: tuple[Tensor, Tensor, Tensor],
    #     hl_node: HLNode,
    #     loss_fn: Callable[[Tensor, Tensor], Tensor],
    # ) -> Tensor:
    #     hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
    #     label_idx = self.get_label_idxs()
    #     # IIT loss is only computed on the tokens we care about
    #     valid_ll_output = (self.weighting*ll_output)[label_idx.as_index][self.mask]
    #     valid_hl_output = (self.weighting*hl_output)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
    #     loss = loss_fn(valid_ll_output, valid_hl_output)
    #     return loss

    # def get_behaviour_loss_over_batch(
    #         self, 
    #         base_input: tuple[Tensor, Tensor, Tensor], 
    #         loss_fn: Callable[[Tensor, Tensor], Tensor]
    #         ) -> Tensor:
    #     base_x, base_y = base_input[0:2]
    #     output = self.ll_model(base_x)
    #     # Apply mask.
    #     label_idx = self.get_label_idxs()
    #     valid_out = (self.weighting*output)[label_idx.as_index][self.mask]
    #     valid_base_y = (self.weighting*base_y)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
    #     behavior_loss = loss_fn(valid_out, valid_base_y)
    #     return behavior_loss
    
    # def get_SIIT_loss_over_batch(
    #         self,
    #         base_input: tuple[Tensor, Tensor, Tensor],
    #         ablation_input: tuple[Tensor, Tensor, Tensor],
    #         loss_fn: Callable[[Tensor, Tensor], Tensor]
    # ) -> Tensor:
    #     base_x, base_y = base_input[0:2]
    #     ablation_x, _ = ablation_input[0:2]
    #     ll_node = self.sample_ll_node()
    #     _, cache = self.ll_model.run_with_cache(ablation_x)
    #     self.ll_cache = cache
    #     out = self.ll_model.run_with_hooks(
    #         base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node))]
    #     )
    #     # print(out.shape, base_y.shape)
    #     label_idx = self.get_label_idxs()
    #     valid_out = (self.weighting*out)[label_idx.as_index][self.mask]
    #     valid_base_y = (self.weighting*base_y)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
    #     siit_loss = loss_fn(valid_out, valid_base_y) 
    #     return siit_loss


    # def run_eval_step(
    #         self, 
    #         base_input: tuple[Tensor, Tensor, Tensor],
    #         ablation_input: tuple[Tensor, Tensor, Tensor],
    #         loss_fn: Callable[[Tensor, Tensor], Tensor]
    #         ) -> dict:
    #     atol = self.training_args["atol"]

    #     # compute IIT loss and accuracy
    #     label_idx = self.get_label_idxs()
    #     hl_node = self.sample_hl_name()
    #     hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
    #     hl_output.to(ll_output.device)
    #     hl_output = hl_output[label_idx.as_index][self.mask]
    #     ll_output = ll_output[label_idx.as_index][self.mask]
    #     if self.hl_model.is_categorical():
    #         loss = loss_fn(ll_output, hl_output)
    #         if ll_output.shape == hl_output.shape:
    #             # To handle the case when labels are one-hot
    #             hl_output = torch.argmax(hl_output, dim=-1)
    #         top1 = torch.argmax(ll_output, dim=-1)
    #         accuracy = (top1 == hl_output).float().mean()
    #         IIA = accuracy.item()
    #     else:
    #         loss = loss_fn(ll_output, hl_output)
    #         IIA = ((ll_output - hl_output).abs() < atol).float().mean().item()

    #     # compute behavioral accuracy
    #     base_x, base_y = base_input[0:2]
    #     output = self.ll_model(base_x)
    #     output = output[label_idx.as_index][self.mask]
    #     base_y = base_y[label_idx.as_index][self.mask]
        
    #     #TODO: how to apply weighting here?
    #     if self.hl_model.is_categorical():
    #         top1 = torch.argmax(output, dim=-1)
    #         if output.shape == base_y.shape:
    #             # To handle the case when labels are one-hot
    #             # TODO: is there a better way?
    #             base_y = torch.argmax(base_y, dim=-1)
    #         accuracy = (top1 == base_y).float().mean()
    #     else:
    #         accuracy = ((output - base_y).abs() < atol).float().mean()    
    #     base_x, base_y = base_input[0:2]
    #     ablation_x, ablation_y = ablation_input[0:2]
        
    #     _, cache = self.ll_model.run_with_cache(ablation_x)
    #     label_idx = self.get_label_idxs()
    #     base_y = base_y[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
    #     self.ll_cache = cache
    #     accuracies = []
    #     for node in self.nodes_not_in_circuit:
    #         out = self.ll_model.run_with_hooks(
    #             base_x, fwd_hooks=[(node.name, self.make_ll_ablation_hook(node))]
    #         )
    #         ll_output = out[label_idx.as_index][self.mask]
    #         if self.hl_model.is_categorical():
    #             if ll_output.shape == base_y.shape:
    #                 base_y = torch.argmax(base_y, dim=-1)
    #             top1 = torch.argmax(ll_output, dim=-1)
    #             accuracy = (top1 == base_y).float().mean().item()
    #         else:
    #             accuracy = ((ll_output - base_y).abs() < self.training_args["atol"]).float().mean().item()
    #         accuracies.append(accuracy)

    #     if len(accuracies) > 0:
    #         accuracy = float(np.mean(accuracies))
    #     else:
    #         accuracy = 1.0

    #     return {
    #         "val/iit_loss": loss.item(),
    #         "val/IIA": IIA,
    #         "val/accuracy": accuracy.item(),
    #         "val/strict_accuracy": accuracy,
    #     }