import torch as t

from poly_bench.cases.duplicate_remover import HighLevelDuplicateRemover
from poly_bench.cases.left_greater import HighLevelLeftGreater
from poly_bench.cases.paren_checker import HighLevelParensBalanceChecker
from poly_bench.cases.unique_extractor import HighLevelUniqueExtractor


def test_HL_duplicate_remover_components():
    # parens balance check
    tokens = [
        "BOS a a b c a b PAD PAD",
        "BOS a b c c c c c c",
        "BOS a b c PAD PAD PAD PAD PAD",
    ]
    tokens = [
        [0, 2, 2, 3, 4, 2, 3, 1, 1],
        [0, 2, 3, 4, 4, 4, 4, 4, 4],
        [0, 2, 3, 4, 1, 1, 1, 1, 1],
    ]
    true_prev_tokens = [[1] + t[:-1] for t in tokens]
    true_equal = [[a == b for a, b in zip(e, p)] for e, p in zip(tokens, true_prev_tokens)]
    true_output = [[1 if eq else a for a, eq in zip(tokens[i], true_equal[i])] for i in range(len(tokens))]

    tokens = t.Tensor(tokens).to(int)
    true_prev_tokens = t.Tensor(true_prev_tokens).to(int)
    true_equal = t.Tensor(true_equal).to(int)
    true_output = t.Tensor(true_output).to(int)
    
    checker = HighLevelDuplicateRemover()
    _, cache   = checker.run_with_cache((tokens, None, None))
    assert t.allclose(cache['prev_token_hook'], true_prev_tokens)
    assert t.allclose(cache['prev_equal_hook'], true_equal)
    assert t.allclose(cache['output_hook'], true_output)


def test_HL_left_greater_components():
    # parens balance check
    tokens = [
        [0, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1],
        [0, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1],
        [0, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
        [0, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        [0, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3],
    ]
    true_lefts = [
        [ 0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3],
    ]
    true_rights = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7],
    ]
    true_mlp0_check = [ # left > right
        [ False,  True, False,  True, False,  True, False, False, False, False, False],
        [ False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [ False, False, False, False, False, False, False, False, False, False, False],
        [ False,  True, False,  True, False,  True, False,  True, False,  True, False],
        [ False, False, False, False, False, False, False, False, False, False, False],
    ]
    true_output  = [ # 2 in the first index, otherwise the integer version of true_mlp0_check
        [ 2, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [ 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_parens = t.stack([true_lefts, true_rights])
    true_mlp0_check = t.Tensor(true_mlp0_check).to(bool)
    true_output = t.nn.functional.one_hot(t.Tensor(true_output).to(int), num_classes=4).float()

    checker = HighLevelLeftGreater()
    output, cache   = checker.run_with_cache((tokens, None, None))
    assert t.allclose(cache['paren_counts_hook'], true_parens)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(output.cpu(), true_output)


def test_HL_parens_balancer_components():
    # parens balance check
    tokens = [
        [0, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1],
        [0, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1],
        [0, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
        [0, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        [0, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3],
    ]
    true_lefts = [
        [ 0,  1,  1,  2,  2,  3,  3,  3,  3,  3,  3],
        [ 0,  1,  2,  3,  4,  5,  5,  5,  5,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3],
    ]
    true_rights = [
        [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  3,  3],
        [ 0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  3],
        [ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5],
        [ 0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7],
    ]
    true_mlp0_check = [ #elevations
        [ 0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [ 0,  1,  2,  3,  4,  5,  4,  3,  2,  2,  2],
        [ 0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0],
        [ 0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0],
        [ 0, -1, -2, -1, -2, -3, -2, -3, -4, -3, -4],
    ]
    horizon_check = [ #first element is ele; rest are horizon
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False,  True, False,  True, False,  True, False,  True, False, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
    ]
    elevation_check = [ #True where mlp0_check is 0, false otherwise.
        [ True, False,  True, False,  True, False,  True,  True,  True,  True,  True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True, False,  True, False,  True, False,  True, False,  True, False,  True],
        [ True, False,  True, False,  True, False,  True, False,  True, False,  True],
        [ True, False, False, False, False, False, False, False, False, False, False],
    ]
    true_mlp1_check = [
        elevation_check, horizon_check
    ]
    true_hor_lookback = [ # For each example, this is False as soon as horizon_check is false once.
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True],
        [ True, False, False, False, False, False, False, False, False, False, False],
        ]
    true_output  = [ # boolean product of true_hor_lookback and elevation_check
        [ 2, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

    tokens = t.Tensor(tokens).to(int)
    true_lefts = t.Tensor(true_lefts).to(int)
    true_rights = t.Tensor(true_rights).to(int)
    true_parens_counts = t.stack((true_lefts, true_rights))
    true_mlp0_check = t.Tensor(true_mlp0_check).to(int)
    true_mlp1_check = t.Tensor(true_mlp1_check).to(bool)
    true_hor_lookback = t.Tensor(true_hor_lookback).to(bool)
    true_output = t.Tensor(true_output).to(int)
    

    checker = HighLevelParensBalanceChecker()
    _, cache   = checker.run_with_cache((tokens, None, None))
    # print(cache['right_parens_hook'] - true_rights)
    assert t.allclose(cache['paren_counts_hook'], true_parens_counts)
    assert t.allclose(cache['mlp0_hook'], true_mlp0_check)
    assert t.allclose(cache['mlp1_hook'], true_mlp1_check)
    assert t.allclose(cache['horizon_lookback_hook'], true_hor_lookback)
    assert t.allclose(cache['mlp2_hook'], true_output)


def test_HL_unique_extractor_components():
    """
    cases:
        "BOS a a b c a b PAD PAD",
        "BOS a b c c c c c c",
        "BOS a b c PAD PAD PAD PAD PAD",
    """
    # parens balance check
    
    tokens = [
        [0, 2, 2, 3, 4, 2, 3, 1, 1],
        [0, 2, 3, 4, 4, 4, 4, 4, 4],
        [0, 2, 3, 4, 1, 1, 1, 1, 1],
    ]
    a_counts = [
        [0, 1, 2, 2, 2, 3, 3, 3, 3],
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    b_counts = [
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1],
    ]
    c_counts = [
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 2, 3, 4, 5, 6],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    true_counts = [a_counts, b_counts, c_counts]
    a_appeared = [
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    b_appeared = [
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    c_appeared = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    true_appeared = [ a_appeared, b_appeared, c_appeared ]
    true_mask = [
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    true_output = [
        [0, 2, 1, 3, 4, 1, 1, 1, 1],
        [0, 2, 3, 4, 1, 1, 1, 1, 1],
        [0, 2, 3, 4, 1, 1, 1, 1, 1],
    ]

    tokens = t.tensor(tokens).to(int)
    true_counts = t.tensor(true_counts).to(int)
    true_appeared = t.tensor(true_appeared).to(int)
    true_mask = t.tensor(true_mask).to(int)
    true_output = t.tensor(true_output).to(int)
    
    checker = HighLevelUniqueExtractor()
    _, cache   = checker.run_with_cache((tokens, None, None))
    
    assert t.allclose(cache['counter_head'], true_counts)
    assert t.allclose(cache['appeared_mlp'], true_appeared)
    assert t.allclose(cache['mask_mlp'], true_mask)
    assert t.allclose(cache['output_mlp'], true_output)