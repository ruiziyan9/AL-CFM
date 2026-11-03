"""Tests for CITNP/models/regime_axis/grouping.py (regime keys and slots)."""

import pytest
import torch

from CITNP.models.regime_axis import group_regimes


def build_mask(specs, D=3, seed=0):
    """specs: per task, a list of (target columns, n rows). Rows are shuffled
    per task. Returns mask [B, Nc, D] and the true regime index [B, Nc]
    (position of the spec entry)."""
    g = torch.Generator().manual_seed(seed)
    masks, labels = [], []
    for spec in specs:
        rows, lab = [], []
        for i, (cols, n) in enumerate(spec):
            row = torch.zeros(D, dtype=torch.bool)
            row[list(cols)] = True
            rows += [row] * n
            lab += [i] * n
        perm = torch.randperm(len(rows), generator=g)
        masks.append(torch.stack(rows)[perm])
        labels.append(torch.tensor(lab)[perm])
    return torch.stack(masks), torch.stack(labels)


SPECS = [
    [((), 5), ((0,), 3), ((2,), 2)],  # R_b = 3
    [((), 10)],  # R_b = 1
    [((0, 1), 6), ((1,), 4)],  # R_b = 2, keys 3 and 2
]


def test_shuffled_unequal():
    mask, lab = build_mask(SPECS)
    g = group_regimes(mask)
    # Hand-built reference: slots follow the sorted keys sum_j a_j 2**j.
    # task 0: () -> key 0 -> slot 0, (0,) -> 1 -> slot 1, (2,) -> 4 -> slot 2
    # task 1: () -> slot 0
    # task 2: (1,) -> key 2 -> slot 0, (0, 1) -> key 3 -> slot 1
    spec_to_slot = [[0, 1, 2], [0], [1, 0]]
    want_slot = torch.stack(
        [torch.tensor(spec_to_slot[b])[lab[b]] for b in range(3)]
    )
    assert torch.equal(g.slot, want_slot)
    assert torch.equal(g.n_r, torch.tensor([[5, 3, 2], [10, 0, 0], [4, 6, 0]]))
    assert torch.equal(
        g.slot_present,
        torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool),
    )
    for b in range(3):
        for r in range(3):
            assert torch.equal(g.slot_member[b, r], want_slot[b] == r)
        same = want_slot[b][:, None] == want_slot[b][None, :]
        assert torch.equal(g.same_regime[b], same)


def test_labels_merge_and_conflict():
    mask, lab = build_mask(SPECS)
    ref = group_regimes(mask)
    # Split the observational rows of task 0 into two labels: they merge.
    split = lab.clone()
    obs0 = (lab[0] == 0).nonzero().flatten()
    split[0, obs0[:2]] = 7
    got = group_regimes(mask, context_regime=split)
    for f in ("slot", "same_regime", "slot_member", "slot_present", "n_r"):
        assert torch.equal(getattr(got, f), getattr(ref, f)), f
    # Padding labels (-1) are ignored.
    pad = lab.clone()
    pad[1, :3] = -1
    group_regimes(mask, context_regime=pad)
    # One label covering two target sets is an error naming the task.
    bad = lab.clone()
    bad[2, (lab[2] == 0).nonzero()[0]] = 1
    with pytest.raises(ValueError, match="task 2"):
        group_regimes(mask, context_regime=bad)


def test_multi_target():
    mask, _ = build_mask([[((0, 2), 3), ((0,), 2), ((2,), 4)]])
    g = group_regimes(mask)
    assert int(g.slot_present.sum()) == 3
    # keys: (0,) -> 1, (2,) -> 4, (0, 2) -> 5
    assert g.n_r.tolist() == [[2, 4, 3]]
    multi = mask[0].all(-1) | (mask[0][:, 0] & mask[0][:, 2])
    assert (g.slot[0][multi] == 2).all()


def test_row_valid_padding():
    mask, _ = build_mask([[((), 3), ((1,), 3)]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    valid[0, :2] = False
    g = group_regimes(mask, context_row_valid=valid)
    assert (g.slot[0, :2] == -1).all() and (g.slot[0, 2:] >= 0).all()
    assert int(g.n_r.sum()) == 4
    assert not g.slot_member[0, :, :2].any()
    # A padded row attends only to itself; no real row attends to it.
    assert torch.equal(g.same_regime[0, :2], torch.eye(6, dtype=torch.bool)[:2])
    assert not g.same_regime[0, 2:, :2].any()


def test_errors():
    with pytest.raises(ValueError, match="at most 62"):
        group_regimes(torch.zeros(1, 4, 63, dtype=torch.bool))
    with pytest.raises(ValueError, match="no valid context row"):
        group_regimes(
            torch.zeros(2, 4, 3, dtype=torch.bool),
            context_row_valid=torch.tensor([[True] * 4, [False] * 4]),
        )
