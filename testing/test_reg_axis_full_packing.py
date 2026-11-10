"""Tests for CITNP/models/regime_axis_full/packing.py (rows packed by regime)."""

import sys
from pathlib import Path

import torch

from CITNP.models.regime_axis import group_regimes
from CITNP.models.regime_axis_full import gather_keys, pack_regimes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reg_axis_grouping import SPECS, build_mask

# Unequal regime sizes and regime counts (R_b = 3, 1, 2, 4), Nc = 12, D = 4.
UNEQUAL = [
    [((), 5), ((0,), 6), ((2,), 1)],
    [((1,), 12)],
    [((0, 1), 3), ((), 9)],
    [((), 1), ((0,), 2), ((1,), 3), ((3,), 6)],
]


def loop_packing(grouping):
    """Per-task, per-regime Python reference: rows of regime r in row order."""
    B, R = grouping.slot_present.shape
    N = int(grouping.n_r.max())
    index = torch.zeros(B, R, N, dtype=torch.long)
    valid = torch.zeros(B, R, N, dtype=torch.bool)
    for b in range(B):
        for r in range(R):
            rows = [i for i, s in enumerate(grouping.slot[b].tolist()) if s == r]
            for k, i in enumerate(rows):
                index[b, r, k] = i
                valid[b, r, k] = True
    return index, valid


def check(grouping):
    packing = pack_regimes(grouping)
    index, valid = loop_packing(grouping)
    assert torch.equal(packing.valid, valid)
    assert torch.equal(packing.index, index)
    assert torch.equal(packing.valid.sum(-1), grouping.n_r)
    assert packing.num_slots == grouping.num_slots
    assert packing.max_rows == int(grouping.n_r.max())
    return packing


def test_packing_equals_loop():
    for specs, D, seed in ((UNEQUAL, 4, 0), (UNEQUAL, 4, 1), (SPECS, 3, 2)):
        mask, _ = build_mask(specs, D=D, seed=seed)  # rows shuffled per task
        packing = check(group_regimes(mask))
        B, R, N = packing.index.shape
        # every real row appears exactly once
        for b in range(B):
            rows = packing.index[b][packing.valid[b]]
            assert sorted(rows.tolist()) == list(range(mask.shape[1]))


def test_key_mask_and_own():
    mask, _ = build_mask(UNEQUAL, D=4)
    grouping = group_regimes(mask)
    packing = pack_regimes(grouping)
    present = grouping.slot_present
    assert not present.all()  # the batch has absent regimes
    assert packing.key_mask.any(-1).all()  # no all-False stage-1 mask row
    assert torch.equal(packing.key_mask[present], packing.valid[present])
    assert packing.key_mask[~present].all()
    assert torch.equal(packing.slot_present, present)
    # own[b, i, r]: regime r is the regime of row i
    assert torch.equal(packing.own.sum(-1), torch.ones_like(grouping.slot))
    assert torch.equal(packing.own.long().argmax(-1), grouping.slot)
    assert torch.equal(packing.n_r, grouping.n_r)


def test_gather_keys():
    mask, _ = build_mask(UNEQUAL, D=4)
    grouping = group_regimes(mask)
    packing = pack_regimes(grouping)
    B, Nc, D = mask.shape
    tokens = torch.randn(B, Nc, D, 5)
    keys = gather_keys(tokens, packing.index)  # [B, R, N, D, d]
    assert keys.shape == (B, *packing.index.shape[1:], D, 5)
    for b in range(B):
        for r in range(packing.num_slots):
            rows = (grouping.slot[b] == r).nonzero().flatten()
            assert torch.equal(keys[b, r, : len(rows)], tokens[b, rows])


def test_padded_rows_belong_to_no_regime():
    mask, _ = build_mask(UNEQUAL, D=4)
    row_valid = torch.ones(mask.shape[:2], dtype=torch.bool)
    row_valid[0, [1, 7]] = False
    row_valid[3, 0] = False
    grouping = group_regimes(mask, context_row_valid=row_valid)
    packing = check(grouping)
    assert int(packing.valid.sum()) == int(row_valid.sum())
    assert not packing.own[0, 1].any() and not packing.own[3, 0].any()


def test_single_regime():
    mask = torch.zeros(2, 6, 3, dtype=torch.bool)
    packing = check(group_regimes(mask))
    assert packing.index.shape == (2, 1, 6) and packing.valid.all()
    assert torch.equal(packing.index[0, 0], torch.arange(6))
