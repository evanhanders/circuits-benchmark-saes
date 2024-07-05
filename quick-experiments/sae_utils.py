import os
import json
from safetensors.torch import save_file
from typing import Optional

import torch as t
import wandb
from datasets import Dataset

import sae_lens
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE_CFG_PATH, SAE_WEIGHTS_PATH, SPARSITY_PATH
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from sae_lens import SAEConfig, SAE, TrainingSAEConfig, TrainingSAE, ActivationsStore, CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig
from sae_lens.training.sae_trainer import SAETrainer

def save_checkpoint(
        trainer: SAETrainer,
        checkpoint_name: int | str,
        wandb_aliases: list[str] | None = None,
    ) -> str:
        """ Lightly modified from https://github.com/jbloomAus/SAELens/blob/v3.5.0/sae_lens/sae_training_runner.py#L161C5-L210C31 """

        sae = trainer.sae
        os.makedirs(trainer.cfg.checkpoint_path, exist_ok=True)
        checkpoint_path = f"{trainer.cfg.checkpoint_path}/{checkpoint_name}"

        os.makedirs(checkpoint_path, exist_ok=True)

        path = f"{checkpoint_path}"
        os.makedirs(path, exist_ok=True)

        if sae.cfg.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(config, f)
        if trainer.cfg.log_to_wandb:
            print(f'saving {path}')
            wandb.save(path)


        log_feature_sparsities = {"sparsity": trainer.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if trainer.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            model_artifact = wandb.Artifact(
                f"{sae.get_name()}",
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )

            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")

            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{sae.get_name()}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

        return checkpoint_path

class RepeatActivationsStore(ActivationsStore):

    def get_batch_tokens(self, batch_size: int | None = None):
        """
        Streams a batch of tokens from a dataset.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                #shuffle self.dataset and restart
                self.iterable_sequences = self._iterate_tokenized_sequences()
                sequences.append(next(self.iterable_sequences))
                # self.iterable_dataset = iter(self.dataset)
                # s = next(self.iterable_dataset)[self.tokens_column]
            
        return t.stack(sequences, dim=0).to(self.model.W_E.device)
    
    def _get_next_dataset_tokens(self) -> t.Tensor:
        device = self.device
        if not self.is_dataset_tokenized:
            try:
                s = next(self.iterable_dataset)[self.tokens_column]
            except StopIteration:
                #shuffle self.dataset and restart
                self.iterable_dataset = iter(self.dataset)
                s = next(self.iterable_dataset)[self.tokens_column]
            tokens = (
                self.model.to_tokens(
                    s,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=self.prepend_bos,
                )
                .squeeze(0)
                .to(device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            try:
                s = next(self.iterable_dataset)[self.tokens_column]
            except StopIteration:
                #shuffle self.dataset and restart
                self.iterable_dataset = iter(self.dataset)
                s = next(self.iterable_dataset)[self.tokens_column]
            tokens = t.tensor(
                s,
                dtype=t.long,
                device=device,
                requires_grad=False,
            )
            if (
                not self.prepend_bos
                and tokens[0] == self.model.tokenizer.bos_token_id  # type: ignore
            ):
                tokens = tokens[1:]
        self.n_dataset_processed += 1
        return tokens


def make_sae_lens_config(
    model : HookedTransformer,
    hook_name: str, 
    hook_layer: int, 
    l1_coeff: float, 
    training_tokens: int = 1_500_000,
    device : str = 'cpu',
    checkpoint_path : str = f"$HOME/persistent-storage/tracr_saes/sae_checkpoints",
    **kwargs
) -> LanguageModelSAERunnerConfig:
    
    cfg_input = dict(
        # Data Generating Function (Model + Training Distribution)
        model_name = "case3",
        model_class_name = "HookedTransformer",
        hook_name = hook_name,
        hook_eval = "NOT_IN_USE",
        hook_layer = hook_layer,
        hook_head_index = None,
        dataset_path = "",
        dataset_trust_remote_code = False,
        streaming = False,
        is_dataset_tokenized = True,
        context_size = 5,
        use_cached_activations = False,
        cached_activations_path = None,  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_head_index}"
    
        # SAE Parameters
        d_in = model.cfg.d_model,
        d_sae = None,
        b_dec_init_method = "geometric_median",
        expansion_factor = 4,
        activation_fn = "relu",  # relu, tanh-relu
        normalize_sae_decoder = True,
        noise_scale = 0.0,
        from_pretrained_path = None,
        apply_b_dec_to_input = False,
        decoder_orthogonal_init = False,
        decoder_heuristic_init = False,
        init_encoder_as_decoder_transpose = False,
    
        # Activation Store Parameters
        training_tokens = training_tokens,
        finetuning_tokens = 0,
        store_batch_size_prompts = 4,
        normalize_activations = "none",  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
    
        # Misc
        device = device,
        act_store_device = "with_model",  # will be set by post init if with_model
        seed = 42,
        dtype = "float32",  # type: ignore #
        prepend_bos = False,
    
        # Performance - see compilation section of lm_runner.py for info
        autocast = False,  # autocast to autocast_dtype during training
        autocast_lm = False,  # autocast lm during activation fetching
        compile_llm = False,  # use torch.compile on the LLM
        llm_compilation_mode = None,  # which torch.compile mode to use
        compile_sae = False,  # use torch.compile on the SAE
        sae_compilation_mode = None,
    
        # Training Parameters
    
        ## Batch size
        train_batch_size_tokens = 320//4,
    
        ## Adam
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
    
        ## Loss Function
        mse_loss_normalization = None,
        l1_coefficient = l1_coeff,
        lp_norm = 1,
        scale_sparsity_penalty_by_decoder_norm = False,
        l1_warm_up_steps = 0,
    
        ## Learning Rate Schedule
        lr = 3e-4,
        lr_scheduler_name = "constant",  # constant, cosineannealing, cosineannealingwarmrestarts
        lr_warm_up_steps = 0,
        lr_end = None,  # only used for cosine annealing, default is lr / 10
        lr_decay_steps = 0,
        n_restart_cycles = 1,  # used only for cosineannealingwarmrestarts
    
        ## FineTuning
        finetuning_method = None,  # scale, decoder or unrotated_decoder
    
        # Resampling protocol args
        use_ghost_grads = True,  # want to change this to true on some timeline.
        feature_sampling_window = 2000,
        dead_feature_window = 1000,  # unless this window is larger feature sampling,
        dead_feature_threshold = 1e-8,
    
        # Evals
        n_eval_batches = 10,
        eval_batch_size_prompts = None,  # useful if evals cause OOM
    
        # WANDB
        log_to_wandb = True,
        log_activations_store_to_wandb = False,
        log_optimizer_state_to_wandb = False,
        wandb_project = "benchmark_saes",
        wandb_id = None,
        run_name = None,
        wandb_entity = None,
        wandb_log_frequency = 10,
        eval_every_n_wandb_logs = 100000000000, # Make this a really big number; currently fails because it tries to compute CE loss.
        # Misc
        resume = False,
        n_checkpoints = 5,
        checkpoint_path = checkpoint_path,
        verbose = True,
        model_kwargs = dict(),
        model_from_pretrained_kwargs = dict(),
        sae_lens_version = str(sae_lens.__version__),
        sae_lens_training_version = str(sae_lens.__version__),
    )
    
    for k, item in kwargs.items():
        if k in cfg_input.keys():
            print(k, item)
            cfg_input[k] = item
        else:
            raise KeyError(f"{k} is not a valid key of an SAELens config")

    return LanguageModelSAERunnerConfig(**cfg_input)

def train_sae(
    model : HookedTransformer,
    runner_cfg : LanguageModelSAERunnerConfig,
    dataset : Dataset,
    
):
    store = RepeatActivationsStore.from_config(model, runner_cfg, dataset=dataset)
    sae = TrainingSAE(runner_cfg)
    trainer = SAETrainer(model, sae, store, save_checkpoint, cfg = runner_cfg)
    
    if runner_cfg.log_to_wandb:
        wandb.init(
            project=runner_cfg.wandb_project,
            config=runner_cfg,
            name=runner_cfg.run_name,
            id=runner_cfg.wandb_id,
        )
    trainer.fit()

    if runner_cfg.log_to_wandb:
        wandb.finish()

    return sae, store
    