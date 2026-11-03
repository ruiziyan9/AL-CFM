"""Regime keys and slots.

A regime is the set of context rows that share one intervention-target set
(row i of `context_int_mask`). Intervention values never matter. Keys and
slots are bookkeeping only: they decide which rows may attend to each other
and which memory slot a row is pooled into, and are never embedded.

Notation: B tasks (or task x query group), Nc context rows, D variables,
R = R_max regimes in the largest task.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

MAX_KEY_BITS = 62  # int64 bitmask keys; larger D needs torch.unique(dim=0) rows


@dataclass(frozen=True)
class RegimeGrouping:
    """Regime structure of a batch.

    slot:         [B, Nc] long, compact regime index of each row, -1 on padded
                  rows. Slot order is the sorted order of the regime keys.
    same_regime:  [B, Nc, Nc] bool, True = rows i and j share a regime (the
                  sample-attention mask). A padded row is True only on its own
                  diagonal entry (it attends to itself, so it stays finite).
    slot_member:  [B, R, Nc] bool, row i belongs to slot r.
    slot_present: [B, R] bool, slot r exists in task b.
    n_r:          [B, R] long, number of real rows in slot r (0 if absent).
    """

    slot: Tensor
    same_regime: Tensor
    slot_member: Tensor
    slot_present: Tensor
    n_r: Tensor

    @property
    def num_slots(self) -> int:
        """R_max."""
        return self.slot_present.shape[1]


def regime_keys(context_int_mask: Tensor) -> Tensor:
    """[B, Nc, D] bool -> [B, Nc] int64 key sum_j a_ij * 2**j."""
    D = context_int_mask.shape[-1]
    if D > MAX_KEY_BITS:
        raise ValueError(
            f"reg_axis regime keys are int64 bitmasks and support at most "
            f"{MAX_KEY_BITS} variables (got D={D}); larger D needs row-wise "
            "torch.unique(dim=0) keys"
        )
    bits = 2 ** torch.arange(D, device=context_int_mask.device, dtype=torch.long)
    return (context_int_mask.long() * bits).sum(-1)


def _check_labels(keys: Tensor, labels: Tensor, valid: Tensor) -> None:
    """Every (task, label) must map to one key. Labels < 0 are ignored."""
    use = valid & (labels >= 0)
    if not use.any():
        return
    b = torch.arange(keys.shape[0], device=keys.device)[:, None].expand_as(keys)
    pairs = torch.stack([b[use], labels[use]], dim=1)  # [M, 2]
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    k = keys[use]
    big = torch.iinfo(torch.long).max
    kmin = torch.full((len(uniq),), big, device=k.device)
    kmax = torch.full((len(uniq),), -1, device=k.device, dtype=torch.long)
    kmin = kmin.scatter_reduce(0, inv, k, "amin")
    kmax = kmax.scatter_reduce(0, inv, k, "amax")
    bad = (kmin != kmax).nonzero().flatten()
    if len(bad):
        task, label = uniq[bad[0]].tolist()
        raise ValueError(
            f"context_regime label {label} in task {task} covers rows with "
            "different intervention-target sets"
        )


def group_regimes(
    context_int_mask: Tensor,
    context_regime: Optional[Tensor] = None,
    context_row_valid: Optional[Tensor] = None,
) -> RegimeGrouping:
    """Group context rows by intervention-target set.

    Args:
        context_int_mask: [B, Nc, D] bool, True on intervened cells.
        context_regime: optional [B, Nc] long labels (-1 = padding), used only
            as a consistency check: each label must cover one target set
            (several labels may share a set; they merge into one slot).
        context_row_valid: optional [B, Nc] bool, False on padded rows (default
            all True). Padded rows belong to no slot.

    Returns:
        RegimeGrouping. Vectorised over rows and tasks (no Python loops).
    """
    B, Nc, _ = context_int_mask.shape
    dev = context_int_mask.device
    keys = regime_keys(context_int_mask)  # [B, Nc]
    if context_row_valid is None:
        valid = torch.ones(B, Nc, dtype=torch.bool, device=dev)
    else:
        valid = context_row_valid.to(torch.bool)
    if context_regime is not None:
        _check_labels(keys, context_regime, valid)

    # Sort distinct (task, key) pairs; padded rows get task index B (dropped).
    b = torch.arange(B, device=dev)[:, None].expand(B, Nc)
    b = torch.where(valid, b, torch.full_like(b, B))
    pairs = torch.stack([b.reshape(-1), keys.reshape(-1)], dim=1)  # [B*Nc, 2]
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    counts = torch.bincount(uniq[:, 0], minlength=B + 1)  # regimes per task
    start = counts.cumsum(0) - counts
    rank = torch.arange(len(uniq), device=dev) - start[uniq[:, 0]]
    slot = rank[inv].view(B, Nc)
    slot = torch.where(valid, slot, torch.full_like(slot, -1))

    R_b = counts[:B]
    if (R_b == 0).any():
        task = int((R_b == 0).nonzero()[0])
        raise ValueError(f"task {task} has no valid context row")
    R = int(R_b.max())

    r = torch.arange(R, device=dev)
    slot_member = slot[:, None, :] == r[None, :, None]  # [B, R, Nc]
    slot_present = r[None, :] < R_b[:, None]  # [B, R]
    n_r = slot_member.sum(-1)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(Nc, dtype=torch.bool, device=dev)[None]
    same_regime = ((slot[:, :, None] == slot[:, None, :]) & pair_valid) | eye
    return RegimeGrouping(
        slot=slot,
        same_regime=same_regime,
        slot_member=slot_member,
        slot_present=slot_present,
        n_r=n_r,
    )
