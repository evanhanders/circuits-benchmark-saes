import pathlib
from typing import Optional, Tuple

import yaml # type: ignore
import json
from huggingface_hub import upload_folder, hf_hub_download # type: ignore
from transformer_lens import HookedTransformer, HookedTransformerConfig # type: ignore
from safetensors.torch import save_file, load_file

from poly_bench.cases import PolyCase, PolyBenchDataset, str_to_model_dict, str_to_dataset_dict
from poly_bench.poly_hl_model import PolyHLModel

def save_model_to_dir(model: HookedTransformer, local_dir: str) -> pathlib.Path:
    directory = pathlib.Path(local_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), directory / 'model.safetensors')
    config_dict = {}
    keys = ['d_model', 'n_layers', 'd_vocab', 'n_ctx', 'd_head', 'seed', 'act_fn']
    for k in keys:
        config_dict[k] = getattr(model.cfg, k)
    with open(directory / 'config.yaml', 'w') as f:
        yaml.dump(config_dict, f)
    return directory


def save_to_hf(local_dir: str, message: str, repo_name: str = 'evanhanders/polysemantic-benchmarks'):
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_name,
        commit_message=message
    )

def load_from_hf(
        model_name: str, 
        model_file: str = 'model.safetensors', 
        config_file : str = 'config.yaml',
        benchmarks_file: Optional[str] = None,
        repo_name: str = 'evanhanders/polysemantic-benchmarks',
        hl_kwargs: dict = {}
        ) -> Tuple[HookedTransformer, PolyCase, PolyBenchDataset | list[PolyBenchDataset]]:
    model_tensors_file = hf_hub_download(repo_id=repo_name, filename=f'{model_name}/{model_file}')
    model_config_file = hf_hub_download(repo_id=repo_name, filename=f'{model_name}/{config_file}')
    if benchmarks_file is not None:
        model_benchmarks_file = hf_hub_download(repo_id=repo_name, filename=f'{model_name}/{benchmarks_file}')
    

    #Get LL Model
    with open(model_config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = HookedTransformerConfig(**config_dict)
    ll_model = HookedTransformer(cfg)
    ll_model.load_state_dict(load_file(model_tensors_file))

    #Get HL Model and dataset class(es)
    if benchmarks_file is not None:
        with open(model_benchmarks_file, 'r') as f:
            model_cases = json.load(f)
        hl_model_cases = [str_to_model_dict[case] for case in model_cases]
        dataset: PolyBenchDataset | list[PolyBenchDataset] = [str_to_dataset_dict[case] for case in model_cases]
        hl_model = PolyHLModel(hl_classes=hl_model_cases, **hl_kwargs)
    else:

        if model_name not in str_to_model_dict or model_name not in str_to_dataset_dict:
            raise ValueError(f"Model name {model_name} not in available models or datasets.")
        hl_model = str_to_model_dict[model_name]()
        dataset = str_to_dataset_dict[model_name]

    return ll_model, hl_model, dataset

def save_poly_model_to_dir(ll_model: HookedTransformer, hl_model, local_dir: str) -> pathlib.Path:
    directory = save_model_to_dir(ll_model, local_dir)
    model_cases = [str(model) for model in hl_model.hl_models]
    with open(directory / 'cases.json', 'w') as f:
        json.dump(model_cases, f)
    return directory