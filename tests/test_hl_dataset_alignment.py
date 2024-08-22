import torch as t
import pytest

from poly_bench.cases.duplicate_remover import HighLevelDuplicateRemover, DuplicateRemoverDataset
from poly_bench.cases.left_greater import HighLevelLeftGreater, LeftGreaterDataset
from poly_bench.cases.paren_checker import HighLevelParensBalanceChecker, BalancedParensDataset
from poly_bench.cases.unique_extractor import HighLevelUniqueExtractor, UniqueExtractorDataset
from poly_bench.poly_hl_model import PolyModelDataset, PolyHLModel

def test_duplicate_remover_dataset_alignment():
    dataset = DuplicateRemoverDataset(N_samples=50, n_ctx=21, seed=42)
    hl_model = HighLevelDuplicateRemover()
    tokens = dataset.get_dataset()['tokens']
    labels = t.tensor(dataset.get_dataset()['labels'])
    hl_output = hl_model((t.tensor(tokens), None, None))
    assert t.allclose(hl_output, labels.to(hl_output.device))

def test_left_greater_dataset_alignment():
    dataset = LeftGreaterDataset(N_samples=50, n_ctx=21, seed=42)
    hl_model = HighLevelLeftGreater()
    tokens = dataset.get_dataset()['tokens']
    labels = t.tensor(dataset.get_dataset()['labels'])
    hl_output = hl_model((t.tensor(tokens), None, None))
    assert t.allclose(hl_output, labels.to(hl_output.device))

def test_paren_checker_dataset_alignment():
    dataset = BalancedParensDataset(N_samples=100, n_ctx=21, seed=42)
    hl_model = HighLevelParensBalanceChecker()
    tokens = dataset.get_dataset()['tokens']
    labels = t.tensor(dataset.get_dataset()['labels'])
    hl_output = hl_model((t.tensor(tokens), None, None))
    assert t.allclose(hl_output, labels.to(hl_output.device))

def test_unique_extractor_dataset_alignment():
    dataset = UniqueExtractorDataset(N_samples=100, n_ctx=21, seed=42)
    hl_model = HighLevelUniqueExtractor()
    tokens = dataset.get_dataset()['tokens']
    labels = t.tensor(dataset.get_dataset()['labels'])
    hl_output = hl_model((t.tensor(tokens), None, None))
    assert t.allclose(hl_output, labels.to(hl_output.device))

dataset_mapping = {
    HighLevelDuplicateRemover : DuplicateRemoverDataset,
    HighLevelLeftGreater : LeftGreaterDataset,
    HighLevelParensBalanceChecker : BalancedParensDataset,
    HighLevelUniqueExtractor : UniqueExtractorDataset
}

@pytest.mark.parametrize('cases', [
    [HighLevelDuplicateRemover, HighLevelLeftGreater],
    [HighLevelDuplicateRemover, HighLevelParensBalanceChecker],
    [HighLevelDuplicateRemover, HighLevelUniqueExtractor],
    [HighLevelLeftGreater, HighLevelParensBalanceChecker],
    [HighLevelLeftGreater, HighLevelUniqueExtractor],
    [HighLevelParensBalanceChecker, HighLevelUniqueExtractor]
])
def test_multi_dataset_alignment(cases):
    dataset_cases = [dataset_mapping[case] for case in cases]
    poly_hl_model = PolyHLModel(hl_classes=cases, size_expansion=1)

    dsets = [dsetcase(N_samples=1000, n_ctx=poly_hl_model.cfg.n_ctx-1) for dsetcase in dataset_cases]
    poly_dataset = PolyModelDataset(dsets, n_ctx=poly_hl_model.cfg.n_ctx)

    input = poly_dataset.get_dataset().inputs
    hl_output = poly_hl_model((input, None, None))
    dataset_label = poly_dataset.get_dataset().targets
    t.allclose(hl_output, dataset_label.to(hl_output.device))
