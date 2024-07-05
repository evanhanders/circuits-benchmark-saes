from datasets import Dataset
from typing import Tuple, Callable
from iit.utils.correspondence import LLNode #Correspondence, HLNode, 

import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

def build_traintest_dataloaders(
    dataset: Dataset, 
    batch_size: int = 256
) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
    train_test_split = dataset.train_test_split(test_size=0.2)
    
    train_t_dataset = TensorDataset(
        t.tensor(train_test_split['train']['tokens']).int(), 
        t.tensor(train_test_split['train']['labels'])
    )
    test_t_dataset = TensorDataset(
        t.tensor(train_test_split['test']['tokens']).int(), 
        t.tensor(train_test_split['test']['labels'])
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

    def ll_ablation_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        keep_mask = t.ones_like(hook_point_out)
        index = ll_node.index if ll_node.index is not None else Ix[[None]]
        keep_mask[index.as_index] -= 1
        if ll_node.subspace is not None:
            subspace = [slice(None)]*(hook_point_out.dim()-1) + [ll_node.subspace]
            keep_mask[tuple(subspace)] -= 1
        hook_point_out = keep_mask*hook_point_out + (1-keep_mask)*ll_cache[hook.name]
        return hook_point_out
       
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

    def get_iit_loss(self, b_input, s_input):
        # sample one of the operations to do the intervention on:
        hl_node = str(self.corr_keys[t.randint(0, len(corr_keys), (1,)).item()])
        ll_nodes = self.corr[hl_node]
    
        #run forward passes to get activations from s
        with t.no_grad():
            _, hl_cache = self.hl_model.run_with_cache(s_input)
            _, ll_cache = self.ll_model.run_with_cache(s_input)
            #run the intervention on the Hl model doing a forward pass with b
            hl_hook_fn = partial(HL_interchange_intervention, cache=hl_cache)
            hl_output = self.hl_model.run_with_hooks(b_input, fwd_hooks=[
                (hl_node.name, hl_hook_fn)
            ])
            hl_label = hl_output[:,-1]
        hooks = []
        for node in ll_nodes:
            # ll_hook_fn = partial(interchange_intervention, cache=ll_cache, key=node.name)
            ll_hook_fn = make_ll_ablation_hook(node, ll_cache)
            hooks.append((node.name, ll_hook_fn))
        ll_output = self.ll_model.run_with_hooks(b_input, fwd_hooks=hooks)
        iit_loss = self.loss_fn(ll_output, hl_label)
        return iit_loss
        

    def train(self, epochs: int, use_wandb: bool = False):
        if use_wandb:
            raise NotImplementedError()
                
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{N}")
            ll_model.train()  # Set the model to training mode
            
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
            train_progress_bar = tqdm(zip(train_dataloader, train_dataloader2), desc="Training", leave=False)
            for b, s in train_progress_bar:
                # Zero the parameter gradients
                optimizer.zero_grad()
        
                ##########
                #IIT loss 
                ##########      
                iit_loss = self.get_iit_loss(b[0], s[0])
        
                ##########
                #SIIT loss 
                ##########
                # Sample a hook from the unused ones
                siit_node = unused_hook_nodes[t.randint(0, len(unused_hook_nodes), (1,)).item()]
                # siit_hook_fn = partial(interchange_intervention, cache=ll_cache, key=siit_key)
                #TODO: Change sampling here.
                siit_hook_fn = make_ll_ablation_hook(siit_node, ll_cache)
                siit_output = ll_model.run_with_hooks(b[0], fwd_hooks=[
                    (siit_node.name, siit_hook_fn)
                ])
                siit_loss = loss_fn(siit_output, b[1].cuda())
        
                ####################
                # Behavior loss
                ####################
                # Get the inputs and labels from the batch
                inputs, labels = b
                outputs = ll_model(inputs)
        
                # Compute the loss
                plain_loss = loss_fn(outputs, labels.cuda())
                loss = plain_weight*plain_loss + iit_loss*iit_weight + siit_loss*siit_weight
        
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
        
                # Update progress bar and accumulate loss
                train_loss += loss.item()
                base_loss += plain_loss.item()
                train_progress_bar.set_postfix(loss=loss.item())
        
            # Evaluation phase
            ll_model.eval()  # Set the model to evaluation mode
            total_loss = 0.0
            with t.no_grad():  # No need to compute gradients during evaluation
                val_progress_bar = tqdm(test_dataloader, desc="Validation", leave=False)
                for batch in val_progress_bar:
                    # Get the inputs and labels from the batch
                    inputs, labels = batch
        
                    # Forward pass
                    outputs = ll_model(inputs)
        
                    # Compute the loss
                    loss = loss_fn(outputs, labels.cuda())
                    total_loss += loss.item()
                    val_progress_bar.set_postfix(loss=loss.item())
        
        
                # Calculate IIA.
                N_samples = 0
                iia_sum = 0
                for b, s in zip(test_dataloader, test_dataloader2):
                    # sample one of the operations to do the intervention on:
                    hl_key = str(corr_keys[t.randint(0, len(corr_keys), (1,)).item()])
                    ll_key = corr_obj[hl_key]
                    
                    #run forward passes to get activations from s
                    _, hl_cache = hl_model.run_with_cache(s[0])
                    _, ll_cache = ll_model.run_with_cache(s[0])
                    
                    #run the intervention on the HL model doing a forward pass with b
                    hl_hook_fn = partial(interchange_intervention, cache=hl_cache, key=hl_key)
                    hl_output = hl_model.run_with_hooks(b[0], fwd_hooks=[
                        (hl_key, hl_hook_fn)
                    ])
                    hl_label = hl_output[:,-1].cuda()
                    #run the intervention on the LL model doing a forward pass with b
                    hooks = []
                    for node in ll_key:
                        # ll_hook_fn = partial(interchange_intervention, cache=ll_cache, key=node.name)
                        ll_hook_fn = make_ll_ablation_hook(node, ll_cache)
                        hooks.append((node.name, ll_hook_fn))
                    ll_output = ll_model.run_with_hooks(b[0], fwd_hooks=hooks)
                    ll_prob = t.sigmoid(ll_output)[:,-1]
        
                    #Calculate similarity of hl_output and ll_output.
                    # This follows eqn 3 of https://arxiv.org/pdf/2112.00826
                    tol = 1e-3 #floating point error tolerance really
                    similarity = t.abs(hl_label - (ll_prob > 0.5).float()) < tol
                    iia_sum += t.sum(similarity).item()
                    N_samples += b[0].shape[0]
                    
            
            avg_iia = iia_sum / N_samples
            avg_train_loss = train_loss / len(train_dataloader)
            avg_val_loss = total_loss / len(test_dataloader)
            avg_base_loss = base_loss / len(train_dataloader)
            print(f"Training loss: ({avg_base_loss:.4f}, {avg_train_loss:.4f}) | Validation loss: {avg_val_loss:.4f} | Validation IIA: {avg_iia}")
            losses.append((avg_train_loss, avg_val_loss))
        print(t.Tensor(losses))
        
        

    