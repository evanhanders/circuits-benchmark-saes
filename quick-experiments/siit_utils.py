from functools import partial
from typing import Tuple, Callable

from datasets import Dataset

import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from iit.utils.correspondence import LLNode #Correspondence, HLNode, 

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

def HL_interchange_intervention(activation, hook, cache):
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
        ll_model, 
        hl_model, 
        dataset, 
        corr,
        unused_nodes,
        loss_fn,
        baseline_weight = 1,
        iit_weight = 1,
        siit_weight = 1,
        batch_size = 256,
        device = 'cpu'
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

    def get_iit_loss(self, b_input, hl_cache, ll_cache):
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

    def get_siit_loss(self, b_input, b_label, ll_cache):
        # Sample a hook from the unused ones
        #TODO: Try different sampling techniques here.
        siit_node = self.unused_nodes[t.randint(0, len(self.unused_nodes), (1,)).item()]

        siit_hook_fn = make_ll_ablation_hook(siit_node, ll_cache)
        siit_output = self.ll_model.run_with_hooks(b_input, fwd_hooks=[
            (siit_node.name, siit_hook_fn)
        ])
        siit_loss = self.loss_fn(siit_output, b_label.float())
        return siit_loss
        

    def train(self, epochs: int, use_wandb: bool = False, **optim_kwargs):
        if use_wandb:
            raise NotImplementedError()

        optimizer = t.optim.AdamW(self.ll_model.parameters(), lr=1e-3, **optim_kwargs)
                
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.ll_model.train()  # Set the model to training mode
            
            # Initialize tqdm progress bar for training
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
            train_progress_bar = tqdm(zip(*self.train_dataloaders), desc="Training", leave=False)
            for b, s in train_progress_bar:
                # Zero the parameter gradients
                optimizer.zero_grad()

                #Run s through to get the activations from it.
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
                siit_loss = self.get_siit_loss(b[0], b[1].to(self.device), ll_cache)
        
                ####################
                # Behavior loss
                ####################
                # Get the inputs and labels from the batch
                inputs, labels = b
                outputs = self.ll_model(inputs)
                baseline_loss = self.loss_fn(outputs, labels.to(self.device))

                
                loss = self.baseline_weight*baseline_loss \
                        + self.iit_weight*iit_loss \
                        + self.siit_weight*siit_loss

                #Update metrics
                metrics['train_loss'].append(loss.item())
                metrics['train_baseline_loss'].append(baseline_loss.item())
                metrics['train_iit_loss'].append(iit_loss.item())
                metrics['train_siit_loss'].append(siit_loss.item())
                metrics['train_IIA'].append(iia.item())
        
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
        
                train_progress_bar.set_postfix(loss=loss.item())
            
        
            # Evaluation phase
            self.ll_model.eval()  # Set the model to evaluation mode
            with t.no_grad():  # No need to compute gradients during evaluation
                
                val_progress_bar = tqdm(zip(*self.test_dataloaders), desc="Validation", leave=False)
                for batch, s in val_progress_bar:
                    
                    inputs, labels = batch
                    
                     #Run s through to get the activations from it.
                    _, hl_cache = self.hl_model.run_with_cache(s[0])
                    _, ll_cache = self.ll_model.run_with_cache(s[0])
                    
                    ##########
                    #IIT loss 
                    ##########      
                    iit_loss, iia = self.get_iit_loss(inputs, hl_cache, ll_cache)
            
                    ##########
                    #SIIT loss 
                    ##########
                    siit_loss = self.get_siit_loss(inputs, labels.to(self.device), ll_cache)
            
                    ####################
                    # Behavior loss
                    ####################
                    # Get the inputs and labels from the batch
        
                    # Forward pass
                    outputs = self.ll_model(inputs)
        
                    # Compute the loss
                    loss = self.loss_fn(outputs, labels.to(self.device))
                    
                    metrics['test_loss'].append(loss.item())
                    metrics['test_baseline_loss'].append(baseline_loss.item())
                    metrics['test_iit_loss'].append(iit_loss.item())
                    metrics['test_siit_loss'].append(siit_loss.item())
                    metrics['test_IIA'].append(iia.item())
                    val_progress_bar.set_postfix(loss=loss.item())

        return metrics
        
        

    