
from typing import Callable 
from jaxtyping import Float

from transformer_lens.hook_points import HookedRootModule
import numpy as np
import torch as t


from iit.model_pairs.ll_model import LLModel
from iit.model_pairs.strict_iit_model_pair import StrictIITModelPair
from iit.utils.correspondence import Correspondence
from iit.utils import index


class ParensModelPair(StrictIITModelPair):

    def __init__(
            self, 
            hl_model: HookedRootModule, 
            ll_model: LLModel, 
            corr: Correspondence, 
            training_args: dict = {}
            ):
        super().__init__(hl_model, ll_model, corr, training_args=training_args)
        default_training_args = {
            "next_token": False,
            "non_ioi_thresh": 0.65,
            "use_per_token_check": False,
        }
        self.training_args = {**default_training_args, **self.training_args}

    @property
    def loss_fn(self) -> Callable[[t.Tensor, t.Tensor], t.Tensor]:

        def per_token_weighted_cross_entropy(
                output: Float[t.Tensor, "batch logit"], 
                target: Float[t.Tensor, "batch logit"] | Float[t.Tensor, "batch"]
                ) -> t.Tensor:
            label_idx = self.get_label_idxs().as_index
            if len(target.shape) == 2: #dumb one-hot fix
                true_target = t.zeros(target.shape[0])
                true_target[target[:, 1] == 1] = 1
            else:
                true_target = target
                
            return t.nn.BCEWithLogitsLoss()(output[label_idx], true_target.to(float).squeeze())

        return per_token_weighted_cross_entropy

    @staticmethod
    def get_label_idxs() -> index.TorchIndex:
        return index.Ix[:, -1]
    
    def get_behaviour_loss_over_batch(
            self, 
            base_input: tuple[t.Tensor, t.Tensor, t.Tensor], 
            loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor]
            ) -> t.Tensor:
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        label_idx = self.get_label_idxs()
        behavior_loss = loss_fn(output[label_idx.as_index], base_y[label_idx.as_index])
        return behavior_loss

    def run_eval_step(
        self,
        base_input: tuple[t.Tensor, t.Tensor, t.Tensor],
        ablation_input: tuple[t.Tensor, t.Tensor, t.Tensor],
        loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor],
    ) -> dict:
        label_idx = self.get_label_idxs().as_index

        # compute IIT loss and accuracy on last token position only
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        hl_argmax = t.argmax(hl_output[:, -1, :], dim=-1)
        loss = loss_fn(ll_output[label_idx], hl_argmax)

        ll_answer = t.round(t.sigmoid(ll_output[:,-1,-1]))
        accuracy = (ll_answer == hl_argmax).float().mean().item()
        IIA = accuracy

        # compute behavioral accuracy
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        ll_answer = t.round(t.sigmoid(output[:,-1,-1]))
        base_accuracy = (ll_answer == base_y.squeeze()).float().mean().item()

        # strict accuracy
        base_x, base_y = base_input[0:2]
        ablation_x, _ = ablation_input[0:2]
        # ll_node = self.sample_ll_node() 
        _, cache = self.ll_model.run_with_cache(ablation_x)
        self.ll_cache = cache
        base_y = base_y[label_idx].to(self.ll_model.cfg.device)
        if self.hl_model.is_categorical:
            if len(base_y.shape) == 2:
                base_y = t.argmax(base_y, dim=-1)
        accuracies = []
        for node in self.nodes_not_in_circuit:
            out = self.ll_model.run_with_hooks(
                base_x, fwd_hooks=[(node.name, self.make_ll_ablation_hook(node))]
            )
            ll_answer = t.round(t.sigmoid(out[:,-1,-1]))
            siit_accuracy = (ll_answer == base_y.squeeze()).float().mean().item()
            accuracies.append(siit_accuracy)
        strict_accuracy = np.mean(accuracies)

        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": base_accuracy,
            "val/strict_accuracy": strict_accuracy,
        }