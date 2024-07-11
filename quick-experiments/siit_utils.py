from functools import partial
from typing import Tuple, Callable
from jaxtyping import Int, Float

from datasets import Dataset

import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint, HookedRootModule
from iit.utils.correspondence import LLNode, Correspondence

def build_traintest_dataloaders(
    dataset: Dataset, 
    batch_size: int = 256,
    device: str = 'cpu'
) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
    train_test_split = dataset.train_test_split(test_size=0.2)
    
    train_t_dataset = TensorDataset(
        t.tensor(train_test_split['train']['tokens']).int(), 
        t.tensor(train_test_split['train']['labels']).float()
    )
    test_t_dataset = TensorDataset(
        t.tensor(train_test_split['test']['tokens']).int(), 
        t.tensor(train_test_split['test']['labels']).float()
    )

    #we need two dataloaders for both train and test to do interchange interventions.
    train_dataloader  = DataLoader(train_t_dataset, batch_size=batch_size, shuffle = True)
    train_dataloader2 = DataLoader(train_t_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader   = DataLoader(test_t_dataset, batch_size=batch_size)
    test_dataloader2  = DataLoader(test_t_dataset, batch_size=batch_size)
    return (train_dataloader, train_dataloader2), (test_dataloader, test_dataloader2)

def HL_interchange_intervention(
    activation : t.Tensor, 
    hook : HookPoint, 
    cache: ActivationCache
) -> t.Tensor:
    activation = cache[hook.name]
    return activation

#lifted from https://github.com/cybershiptrooper/iit/blob/main/iit/model_pairs/base_model_pair.py#L148 and modified
def make_ll_ablation_hook(
    ll_node: LLNode,
    ll_cache: ActivationCache
) -> Callable[[t.Tensor, HookPoint], t.Tensor]:

    # def ll_ablation_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
    #     keep_mask = t.ones_like(hook_point_out)
    #     index = ll_node.index if ll_node.index is not None else Ix[[None]]
    #     keep_mask[index.as_index] -= 1
    #     if ll_node.subspace is not None:
    #         subspace = [slice(None)]*(hook_point_out.dim()-1) + [ll_node.subspace]
    #         keep_mask[tuple(subspace)] -= 1
    #     hook_point_out = keep_mask*hook_point_out + (1-keep_mask)*ll_cache[hook.name]
    #     return hook_point_out
    def ll_ablation_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        out = hook_point_out.clone()
        index = ll_node.index if ll_node.index is not None else Ix[[None]]
        if ll_node.subspace is not None:
            subspace = [slice(None)]*(hook_point_out.dim()-1) + [ll_node.subspace]
            out[index.as_index][tuple(subspace)] = ll_cache[hook.name][index.as_index][tuple(subspace)]
        else:
            out[index.as_index] = ll_cache[hook.name][index.as_index]
        return out
       
    return ll_ablation_hook

def make_post_ablation_hook(
    ll_node: LLNode,
    ll_cache: ActivationCache,
    method: str = "mean"
) -> Callable[[t.Tensor, HookPoint], t.Tensor]:

    def ll_ablation_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        keep_mask = t.ones_like(hook_point_out)
        index = ll_node.index if ll_node.index is not None else Ix[[None]]
        keep_mask[index.as_index] -= 1
        if ll_node.subspace is not None:
            subspace = [slice(None)]*(hook_point_out.dim()-1) + [ll_node.subspace]
            keep_mask[tuple(subspace)] -= 1
        if method == "mean":
            hook_point_out = keep_mask*hook_point_out + (1-keep_mask)*t.mean(ll_cache[hook.name], dim=1, keepdim=True)
        elif method == 'zero':
            hook_point_out = keep_mask*hook_point_out
        else:
            raise ValueError(f"Unknown ablation method: {method}")
        return hook_point_out
       
    return ll_ablation_hook

class ModelTrainerSIIT:
    def __init__(
        self,
        ll_model : HookedTransformer, 
        hl_model : HookedRootModule, 
        dataset : Dataset, 
        corr: Correspondence,
        unused_nodes : list,
        loss_fn : Callable,
        baseline_weight : float = 1.,
        iit_weight : float = 1.,
        siit_weight : float = 1.,
        batch_size : int = 256,
        device : str = 'cpu'
    ):
        self.ll_model = ll_model.to(device)
        self.hl_model = hl_model.to(device)
        self.dataset = dataset
        self.corr = corr
        self.corr_keys = list(corr.keys())
        self.unused_nodes = unused_nodes
        self.loss_fn = loss_fn
        self.train_dataloaders, self.test_dataloaders = build_traintest_dataloaders(dataset, batch_size = batch_size)
        self.device = device
        self.baseline_weight = baseline_weight
        self.iit_weight = iit_weight
        self.siit_weight = siit_weight

    def get_iit_loss(
        self, 
        b_input : Int[t.Tensor, "batch n_ctx"], 
        hl_cache : ActivationCache, 
        ll_cache : ActivationCache
    ) -> Tuple[Float[t.Tensor, ''], Float[t.Tensor, '']]:
        # sample one of the operations to do the intervention on:
        hl_node = self.corr_keys[t.randint(0, len(self.corr_keys), (1,)).item()]
        ll_nodes = self.corr[hl_node]
        
        #run the intervention on the Hl model doing a forward pass with b
        hl_hook_fn = partial(HL_interchange_intervention, cache=hl_cache)
        hl_output = self.hl_model.run_with_hooks(b_input, fwd_hooks=[
            (hl_node.name, hl_hook_fn)
        ])
        hl_label = hl_output[:,-1].to(self.device)
        
        hooks = []
        for node in ll_nodes:
            # ll_hook_fn = partial(interchange_intervention, cache=ll_cache, key=node.name)
            ll_hook_fn = make_ll_ablation_hook(node, ll_cache)
            hooks.append((node.name, ll_hook_fn))
        ll_output = self.ll_model.run_with_hooks(b_input, fwd_hooks=hooks)
        iit_loss = self.loss_fn(ll_output, hl_label)

        #Calculate similarity of hl_output and ll_output.
        # This follows eqn 3 of https://arxiv.org/pdf/2112.00826
        tol = 1e-3 # should be floating point error tolerance really
        ll_prob = t.sigmoid(ll_output)[:,-1,-1]
        similarity = t.abs(hl_label - (ll_prob > 0.5).float()) < tol
        iia = t.sum(similarity) / similarity.shape[0]
        
        return iit_loss, iia

    def get_siit_loss(
        self, 
        b_input : Int[t.Tensor, "batch n_ctx"], 
        b_label : Float[t.Tensor, "batch"], 
        ll_cache : ActivationCache,
        sampling_mode : str = 'sample_all' #or 'individual', 'all'
    ) -> Float[t.Tensor, '']:
        # Sample a hook from the unused ones
        if sampling_mode == 'individual':
            siit_node = self.unused_nodes[t.randint(0, len(self.unused_nodes), (1,)).item()]
            nodes = [siit_node,]
        elif sampling_mode == 'sample_all':
            importance = t.randint(0, 2, (len(self.unused_nodes),)).to(bool).tolist()
            nodes = [node for node, imp in zip(self.unused_nodes, importance) if imp]
        elif sampling_mode == 'all':
            nodes = self.unused_nodes
        else:
            raise ValueError(f"Unexpected SIIT sampling mode: {sampling_mode}")
        hooks = []
        for node in nodes:
            hooks.append((node.name, make_ll_ablation_hook(node, ll_cache)))
        siit_output = self.ll_model.run_with_hooks(b_input, fwd_hooks=hooks)
        siit_loss = self.loss_fn(siit_output, b_label)
        return siit_loss
        

    def train(
        self, 
        epochs: int, 
        use_wandb: bool = False, 
        **optim_kwargs
    ) -> dict:
        if use_wandb:
            raise NotImplementedError()

        optimizer = t.optim.AdamW(self.ll_model.parameters(), lr=1e-3, **optim_kwargs)
        metrics = {
            "train_loss" : [],
            "train_baseline_loss" : [],
            "train_iit_loss" : [],
            "train_siit_loss" : [],
            "test_loss" : [],
            "test_baseline_loss" : [],
            "test_iit_loss" : [],
            "test_siit_loss" : [],
            "train_IIA" : [],
            "test_IIA" : [],
        }
        # Training loop
        epoch_progress_bar = tqdm(range(epochs), desc=f"Epoch 1/{epochs}", leave=True, position=0)
        for epoch in epoch_progress_bar:
            self.ll_model.train()  # Set the model to training mode
            
            
            train_progress_bar = tqdm(zip(*self.train_dataloaders), desc="Training", leave=False, total=len(self.train_dataloaders[0]), position=1)
            for b, s in train_progress_bar:
                optimizer.zero_grad()

                with t.no_grad():
                    _, hl_cache = self.hl_model.run_with_cache(s[0])
                    _, ll_cache = self.ll_model.run_with_cache(s[0])
        
                ##########
                #IIT loss 
                ##########      
                iit_loss, iia = self.get_iit_loss(b[0], hl_cache, ll_cache)
        
                ##########
                #SIIT loss 
                ##########
                siit_loss = self.get_siit_loss(b[0], b[1].float().to(self.device), ll_cache, sampling_mode='all')
        
                ####################
                # Behavior loss
                ####################
                inputs, labels = b
                outputs = self.ll_model(inputs)
                baseline_loss = self.loss_fn(outputs, labels.to(self.device))

                
                loss = self.baseline_weight*baseline_loss \
                        + self.iit_weight*iit_loss \
                        + self.siit_weight*siit_loss

                metrics['train_loss'].append(loss.item())
                metrics['train_baseline_loss'].append(baseline_loss.item())
                metrics['train_iit_loss'].append(iit_loss.item())
                metrics['train_siit_loss'].append(siit_loss.item())
                metrics['train_IIA'].append(iia.item())
        
                loss.backward()
                optimizer.step()
        
                train_progress_bar.set_postfix(loss=f'{loss.item():.2e}', iia=f'{iia.item():.3f}')
                # train_progress_bar.update(1)
            train_progress_bar.close()
            
        
            # Evaluation phase
            self.ll_model.eval() 
            with t.no_grad():  # Don't compute gradients during evaluation
                val_progress_bar = tqdm(zip(*self.test_dataloaders), desc="Testing", leave=False, total=len(self.test_dataloaders[0]), position=1)
                measures = [0]*5
                n_iters = 0
                for batch, s in val_progress_bar:
                    
                    inputs, labels = batch
                    
                    _, hl_cache = self.hl_model.run_with_cache(s[0])
                    _, ll_cache = self.ll_model.run_with_cache(s[0])
                    
                    ##########
                    #IIT loss 
                    ##########      
                    iit_loss, iia = self.get_iit_loss(inputs, hl_cache, ll_cache)
            
                    ##########
                    #SIIT loss 
                    ##########
                    siit_loss = self.get_siit_loss(inputs, labels.float().to(self.device), ll_cache)
            
                    ####################
                    # Behavior loss
                    ####################
                    outputs = self.ll_model(inputs)
                    loss = self.loss_fn(outputs, labels.to(self.device))

                    measures[0] += loss.item()
                    measures[1] += baseline_loss.item()
                    measures[2] += iit_loss.item()
                    measures[3] += siit_loss.item()
                    measures[4] += iia.item()
                    n_iters += 1
                    
                    val_progress_bar.set_postfix(loss=f'{loss.item():.2e}', iia=f'{iia.item():.3f}')
                    # val_progress_bar.update(1)
            
                metrics['test_loss'].append(measures[0] / n_iters)
                metrics['test_baseline_loss'].append(measures[1] / n_iters)
                metrics['test_iit_loss'].append(measures[2] / n_iters)
                metrics['test_siit_loss'].append(measures[3] / n_iters)
                metrics['test_IIA'].append(measures[4] / n_iters)
                val_progress_bar.close()
            # epoch_progress_bar.update(1)
            epoch_progress_bar.set_postfix(test_loss=metrics['test_loss'][-1], test_IIA=metrics['test_IIA'][-1])
            epoch_progress_bar.set_description(f"Epoch {epoch+2}/{epochs}")

        epoch_progress_bar.close()
        return metrics
        
        

    