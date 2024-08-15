from collections import defaultdict
from functools import partial
from typing import Optional, Callable


import torch
import numpy as np
from torch import nn, Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint
from iit.utils.index import TorchIndex, Ix
from iit.utils.correspondence import Correspondence
from iit.utils.nodes import HLNode

from circuits_benchmark.utils.iit.iit_hl_model import IITHLModel
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_PAD


class PolyHLModel(HookedRootModule):

    def __init__(
            self, 
            hl_models: list[IITHLModel], 
            corrs: list[Correspondence], 
            cases = list[TracrBenchmarkCase],
            attn_suffix: str = 'attn.hook_z' #force all attn hooks to have the same suffix
            ):
        super().__init__()
        self.hl_models = hl_models
        self.corrs = corrs
        self.cases = cases

        self.tracr_d_heads = [mod.W_Q.shape[-1] for mod in self.hl_models]
        self.tracr_d_mlps = [mod.W_in[0].shape[1] for mod in self.hl_models]
        self.tracr_d_models = [mod.cfg.d_model for mod in self.hl_models]
        self.tracr_n_ctx = [mod.cfg.n_ctx for mod in self.hl_models]

        self.n_layers = max([mod.cfg.n_layers for mod in self.hl_models])
        self.n_heads = max([mod.cfg.n_heads for mod in self.hl_models])
        self.n_ctx = max(self.tracr_n_ctx) + 1
        self.d_head = max(self.tracr_d_heads)
        self.d_mlp = max(self.tracr_d_mlps)
        self.attn_shapes = []

        #make hooks for each necessary attn head and each mlp
        self.input_hook = HookPoint()
        self.task_hook = HookPoint()
        self.attn_hooks = nn.ModuleList([nn.ModuleList([HookPoint() for _ in range(self.n_heads)]) for _ in range(self.n_layers)])
        self.mlp_hooks = nn.ModuleList([HookPoint() for _ in range(self.n_layers)])

        corr_dict = defaultdict(list)

        for model_number, corr in enumerate(corrs):
            if corr.suffixes['attn'] == 'attn.hook_result':
                self.attn_shapes.append(self.tracr_d_models[model_number])
            else:
                self.attn_shapes.append(self.tracr_d_heads[model_number])
            corr_keys = defaultdict(list)
            for k in corr.keys():
                corr_keys[k.name].append(k)
            for i in range(self.n_layers):
                for k, suff in corr.suffixes.items():

                    name = f'blocks.{i}.{suff}'
                    if k == 'attn':
                        if model_number == 0:
                            for _ in range(self.n_heads):
                                corr_dict[name].append([])
                        if name in corr_keys.keys():
                            hl_nodes = corr_keys[name]
                            for hl_node in hl_nodes:
                                ll_nodes = corr[hl_node]
                                for head in range(self.n_heads):
                                    used = False
                                    for node in ll_nodes:
                                        if node.index.as_index[2] == head:
                                            corr_dict[name][head].append(corr_keys[name])
                                            used = True
                                            break
                                    if not used:
                                        corr_dict[name][head].append(None)
                        else:
                            for head in range(self.n_heads):
                                corr_dict[name][head].append(None)
                    elif k == 'mlp':
                        if name in corr_keys.keys():
                            corr_dict[name].append(corr_keys[name])
                        else:
                            corr_dict[name].append(None)
                    else:
                        raise ValueError(f"Unknown suffix in correspondence: {k}")
        self.attn_shape = max(self.attn_shapes)
        self.corr_prep_dict = corr_dict

        corr_dict = {}
        task_id_set = False
        corr_dict['input_hook'] = [('blocks.0.hook_resid_pre', Ix[[None]], None),]
        for i in range(self.n_layers):
            for j in range(self.n_heads):
                use_attn_head = False
                use_mlp = False
                for k in range(len(self.hl_models)):
                    corr = self.corrs[k]
                    suffixes = corr.suffixes
                    attn_hook_name = f'blocks.{i}.{suffixes["attn"]}'
                    new_attn_hook_name = f'blocks.{i}.{attn_suffix}'
                    mlp_hook_name = f'blocks.{i}.{suffixes["mlp"]}'
                    if self.corr_prep_dict[attn_hook_name][j][k] is not None:
                        use_attn_head = True
                    if self.corr_prep_dict[mlp_hook_name][k] is not None:
                        use_mlp = True
                if use_attn_head:
                    corr_dict[f'attn_hooks.{i}.{j}'] = [(new_attn_hook_name, Ix[[None,None,j,None]], None),]
                elif not task_id_set:
                    corr_dict[f'task_hook'] = [(new_attn_hook_name, Ix[[None,None,j,None]], None),]
                    task_id_set = True
                if use_mlp and j == 0:
                    corr_dict[f'mlp_hooks.{i}'] = [(mlp_hook_name, Ix[[None]], None),]
        self.corr = Correspondence.make_corr_from_dict(corr_dict, suffixes={'attn': attn_suffix, 'mlp': suffixes['mlp']})

        self.setup()
    
    def is_categorical(self):
        return False
    
    def forward(self, x):
        # get sorting indices by task id
        if isinstance(x, tuple):
            x = x[0]
        tokens = self.input_hook(x)
        task_ids = tokens[:, 0]
        task_ids = self.task_hook(task_ids)

        # Step 1 -- get all the activations.
        caches = []
        for i, hl_model in enumerate(self.hl_models):
            if isinstance(hl_model, IITHLModel):
                hl_model = hl_model.hl_model
            #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            encoder = hl_model.tracr_input_encoder
            task_tokens = torch.clone(tokens)[:, 1:hl_model.cfg.n_ctx+1]
            task_tokens[task_tokens >= hl_model.cfg.d_vocab] = encoder.encoding_map[TRACR_PAD]
            _, cache = hl_model.run_with_cache(task_tokens)
            caches.append(cache)
        
        # for k, i in caches[0].items():
        #     print(k, i.shape)
        
        #ActivationCache isn't writable so we need to copy it.
        caches = [{k: i for k, i in cache.items()} for cache in caches]


        # Step 2 -- construct all the hooks for THIS model.
        for layer in range(self.n_layers):
            # create MLP hooks
            mlp_storage = torch.zeros((len(self.hl_models), tokens.shape[0], tokens.shape[1], self.d_mlp))
            attn_storage = torch.zeros((len(self.hl_models), tokens.shape[0], tokens.shape[1], self.n_heads, self.attn_shape))
            for i, hl_model in enumerate(self.hl_models):

                #MLP
                suffix = self.corrs[i].suffixes['mlp']
                hook_name = f'blocks.{layer}.{suffix}'
                #unpack from cache
                if self.corr_prep_dict[hook_name][i] is not None:
                    # print(hook_name, mlp_storage.shape, caches[i][hook_name].shape)
                    mlp_storage[i,:,1:hl_model.cfg.n_ctx+1,:self.tracr_d_mlps[i]] = caches[i][hook_name]
                

                #Attn
                suffix = self.corrs[i].suffixes['attn']
                hook_name = f'blocks.{layer}.{suffix}'
                attn_shape = self.attn_shapes[i]
                #unpack from cache
                for head in range(self.n_heads):
                    if self.corr_prep_dict[hook_name][head][i] is not None:
                        # print(attn_storage[i,:,1:hl_model.cfg.n_ctx+1,head,:attn_shape].shape, caches[i][hook_name][:,:,head].shape)
                        attn_storage[i,:,1:hl_model.cfg.n_ctx+1,head,:attn_shape] = caches[i][hook_name][:,:,head]

            #modify with hook
            mlp_storage = self.mlp_hooks[layer](mlp_storage)
            for head in range(self.n_heads):
                attn_storage[:,:,:,head] = self.attn_hooks[layer][head](attn_storage[:,:,:,head])

                
            for i, hl_model in enumerate(self.hl_models):
                #pack back into cache
                # MLP
                suffix = self.corrs[i].suffixes['mlp']
                hook_name = f'blocks.{layer}.{suffix}'
                caches[i][hook_name] = mlp_storage[i,:,1:hl_model.cfg.n_ctx+1,:self.tracr_d_mlps[i]]
                # Attn
                suffix = self.corrs[i].suffixes['attn']
                hook_name = f'blocks.{layer}.{suffix}'
                attn_shape = self.attn_shapes[i]
                caches[i][hook_name] = attn_storage[i,:,1:hl_model.cfg.n_ctx+1,:hl_model.cfg.n_heads,:attn_shape]


        # Step 3 -- run with a bunch of hooks on the tracr models using the cache, actually doing interventions, and return intervened output.
        outputs = torch.zeros((tokens.shape[0], tokens.shape[1], 1))
        self.mask = torch.ones_like(outputs).to(bool)
        self.mask[:,0] = False
        for i, hl_model in enumerate(self.hl_models):
            if isinstance(hl_model, IITHLModel):
                hl_model = hl_model.hl_model
            #clip token vocab down to task vocab size; replace out-of-task tokens with PAD.
            encoder = hl_model.tracr_input_encoder
            task_tokens = torch.clone(tokens)[:, 1:hl_model.cfg.n_ctx+1]
            task_tokens[task_tokens >= hl_model.cfg.d_vocab] = encoder.encoding_map[TRACR_PAD]

            # define hooks.
            def mlp_replacement_hook(x, hook):
                x[:] = caches[i][hook.name]
            
            def attn_replacement_hook(x, hook, index: Optional[TorchIndex] = None):
                x[index.as_index] = caches[i][hook.name][index.as_index]

            hooks = []
            #go through self.corr_dict and link up each hook with corresponding hook(s) in HL tracr models.
            for layer in range(self.n_layers):

                #MLP
                suffix = self.corrs[i].suffixes['mlp']
                hook_name = f'blocks.{layer}.{suffix}'
                #unpack from cache
                if self.corr_prep_dict[hook_name][i] is not None:
                    hooks.append((hook_name, mlp_replacement_hook))
                
                # Attn
                suffix = self.corrs[i].suffixes['attn']
                hook_name = f'blocks.{layer}.{suffix}'
                for head in range(hl_model.cfg.n_heads):
                    if self.corr_prep_dict[hook_name][head][i] is not None:
                        hooks.append((hook_name, partial(attn_replacement_hook, index=TorchIndex([None,None,head,None]))))

            # print(hooks)
            model_output = hl_model.run_with_hooks(task_tokens, fwd_hooks=hooks)
            outputs[task_ids == i, 1:hl_model.cfg.n_ctx+1,:] = model_output[task_ids == i]
        
            #TODO: Make & return (or store?) a mask for evaluating the loss on the output.
            self.mask[task_ids == i, hl_model.cfg.n_ctx+1:] = False
            self.weighting = max(self.tracr_n_ctx) / self.mask.sum(dim=1, keepdim=True)

        return outputs
    
    def get_IIT_loss_over_batch(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        hl_node: HLNode,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        label_idx = self.get_label_idxs()
        # IIT loss is only computed on the tokens we care about
        valid_ll_output = (self.weighting*ll_output)[label_idx.as_index][self.mask]
        valid_hl_output = (self.weighting*hl_output)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
        loss = loss_fn(valid_ll_output, valid_hl_output)
        return loss

    def get_behaviour_loss_over_batch(
            self, 
            base_input: tuple[Tensor, Tensor, Tensor], 
            loss_fn: Callable[[Tensor, Tensor], Tensor]
            ) -> Tensor:
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        # Apply mask.
        label_idx = self.get_label_idxs()
        valid_out = (self.weighting*output)[label_idx.as_index][self.mask]
        valid_base_y = (self.weighting*base_y)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
        behavior_loss = loss_fn(valid_out, valid_base_y)
        return behavior_loss
    
    def get_SIIT_loss_over_batch(
            self,
            base_input: tuple[Tensor, Tensor, Tensor],
            ablation_input: tuple[Tensor, Tensor, Tensor],
            loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Tensor:
        base_x, base_y = base_input[0:2]
        ablation_x, _ = ablation_input[0:2]
        ll_node = self.sample_ll_node()
        _, cache = self.ll_model.run_with_cache(ablation_x)
        self.ll_cache = cache
        out = self.ll_model.run_with_hooks(
            base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node))]
        )
        # print(out.shape, base_y.shape)
        label_idx = self.get_label_idxs()
        valid_out = (self.weighting*out)[label_idx.as_index][self.mask]
        valid_base_y = (self.weighting*base_y)[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
        siit_loss = loss_fn(valid_out, valid_base_y) 
        return siit_loss


    def run_eval_step(
            self, 
            base_input: tuple[Tensor, Tensor, Tensor],
            ablation_input: tuple[Tensor, Tensor, Tensor],
            loss_fn: Callable[[Tensor, Tensor], Tensor]
            ) -> dict:
        atol = self.training_args["atol"]

        # compute IIT loss and accuracy
        label_idx = self.get_label_idxs()
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        hl_output.to(ll_output.device)
        hl_output = hl_output[label_idx.as_index][self.mask]
        ll_output = ll_output[label_idx.as_index][self.mask]
        if self.hl_model.is_categorical():
            loss = loss_fn(ll_output, hl_output)
            if ll_output.shape == hl_output.shape:
                # To handle the case when labels are one-hot
                hl_output = torch.argmax(hl_output, dim=-1)
            top1 = torch.argmax(ll_output, dim=-1)
            accuracy = (top1 == hl_output).float().mean()
            IIA = accuracy.item()
        else:
            loss = loss_fn(ll_output, hl_output)
            IIA = ((ll_output - hl_output).abs() < atol).float().mean().item()

        # compute behavioral accuracy
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        output = output[label_idx.as_index][self.mask]
        base_y = base_y[label_idx.as_index][self.mask]
        
        #TODO: how to apply weighting here?
        if self.hl_model.is_categorical():
            top1 = torch.argmax(output, dim=-1)
            if output.shape == base_y.shape:
                # To handle the case when labels are one-hot
                # TODO: is there a better way?
                base_y = torch.argmax(base_y, dim=-1)
            accuracy = (top1 == base_y).float().mean()
        else:
            accuracy = ((output - base_y).abs() < atol).float().mean()    
        base_x, base_y = base_input[0:2]
        ablation_x, ablation_y = ablation_input[0:2]
        
        _, cache = self.ll_model.run_with_cache(ablation_x)
        label_idx = self.get_label_idxs()
        base_y = base_y[label_idx.as_index][self.mask].to(self.ll_model.cfg.device)
        self.ll_cache = cache
        accuracies = []
        for node in self.nodes_not_in_circuit:
            out = self.ll_model.run_with_hooks(
                base_x, fwd_hooks=[(node.name, self.make_ll_ablation_hook(node))]
            )
            ll_output = out[label_idx.as_index][self.mask]
            if self.hl_model.is_categorical():
                if ll_output.shape == base_y.shape:
                    base_y = torch.argmax(base_y, dim=-1)
                top1 = torch.argmax(ll_output, dim=-1)
                accuracy = (top1 == base_y).float().mean().item()
            else:
                accuracy = ((ll_output - base_y).abs() < self.training_args["atol"]).float().mean().item()
            accuracies.append(accuracy)

        if len(accuracies) > 0:
            accuracy = float(np.mean(accuracies))
        else:
            accuracy = 1.0

        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": accuracy.item(),
            "val/strict_accuracy": accuracy,
        }