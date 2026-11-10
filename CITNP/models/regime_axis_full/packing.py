"""Packing of context rows by regime for reg_axis_full.

The context arrives flat, [B, Nc] rows in any order. Stage 1 of the regime
attention needs the rows of one regime side by side, so the rows are packed
into [B, R, N]: regime r of task b occupies positions 0..n_r - 1 of
`index[b, r]`, in their original row order. Readers are never packed.

Notation: B tasks (x query groups), Nc context rows, D variables,
d = d_model, R = R_max regimes in the batch, N = max rows of one regime.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from CITNP.models.regime_axis import RegimeGrouping


@dataclass(frozen=True)
class RegimePacking:
    """Context rows packed by regime, built once per forward pass.

    index:        [B, R, N] long, row index of the k-th row of regime r; 0 at
                  padded positions (a valid index, masked by `valid`).
    valid:        [B, R, N] bool, True on real rows; `valid.sum(-1) == n_r`.
    key_mask:     [B, R, N] bool, the stage-1 key mask (True = attend):
                  `valid`, except that every position of an absent regime is
                  True, so no mask row is all False.
    slot_present: [B, R] bool, regime r exists in task b (the stage-2 key
                  mask).
    own:          [B, Nc, R] bool, regime r is the regime of context row i.
    n_r:          [B, R] long, number of rows of regime r (0 if absent).
    """

    index: Tensor
    valid: Tensor
    key_mask: Tensor
    slot_present: Tensor
    own: Tensor
    n_r: Tensor

    @property
    def num_slots(self) -> int:
        """R_max."""
        return self.index.shape[1]

    @property
    def max_rows(self) -> int:
        """N, the largest regime of the batch."""
        return self.index.shape[2]


def pack_regimes(grouping: RegimeGrouping) -> RegimePacking:
    """Build the packing of a batch from its regime grouping.

    Args:
        grouping: output of `group_regimes` (slot [B, Nc], n_r [B, R], ...).

    Returns:
        RegimePacking. Vectorised over tasks and rows (no Python loops).
    """
    slot, n_r = grouping.slot, grouping.n_r
    B, Nc = slot.shape
    R = grouping.num_slots
    N = int(n_r.max())
    dev = slot.device

    # Sort the rows of each task by regime (stable: original order within a
    # regime). Padded rows (slot -1) go to an extra bin R at the end.
    bin_of_row = torch.where(slot < 0, torch.full_like(slot, R), slot)
    order = torch.argsort(bin_of_row, dim=1, stable=True)  # [B, Nc]
    sorted_bin = bin_of_row.gather(1, order)  # [B, Nc]
    # Position of each sorted row within its regime.
    start = n_r.cumsum(1) - n_r  # [B, R] first sorted position of regime r
    start = torch.cat([start, n_r.sum(1, keepdim=True)], dim=1)  # + bin R
    rank = torch.arange(Nc, device=dev)[None] - start.gather(1, sorted_bin)
    # Scatter row indices to (regime, rank); bin R lands beyond R * N and is
    # cut off.
    flat = sorted_bin * N + rank  # [B, Nc], unique per task
    index = torch.zeros(B, R * N + Nc, dtype=torch.long, device=dev)
    index = index.scatter(1, flat, order)[:, : R * N].reshape(B, R, N)

    valid = torch.arange(N, device=dev)[None, None, :] < n_r[:, :, None]
    present = grouping.slot_present
    return RegimePacking(
        index=index,
        valid=valid,
        key_mask=valid | ~present[:, :, None],
        slot_present=present,
        own=grouping.slot_member.transpose(1, 2),
        n_r=n_r,
    )


def gather_keys(tokens: Tensor, index: Tensor) -> Tensor:
    """Gather flat context tokens into their packed positions.

    Args:
        tokens: [B, Nc, D, d] context tokens (any trailing shape works).
        index: [B, R, N] long row indices (`RegimePacking.index`).

    Returns:
        [B, R, N, D, d]; entry [b, r, k] is tokens[b, index[b, r, k]]. Padded
        positions hold a copy of row 0 and must be masked.
    """
    batch = torch.arange(tokens.shape[0], device=tokens.device)[:, None, None]
    return tokens[batch, index]
