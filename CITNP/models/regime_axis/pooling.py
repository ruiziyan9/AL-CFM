"""Regime memories: attention pooling (PMA) of each regime into K tokens.

Notation: B tasks, Nc context rows, D variables, d = d_model, K memory tokens
per (regime, variable), R = R_max regime slots.
"""

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.regime_axis.attention import MaskedSetAttentionLayer
from CITNP.models.regime_axis.grouping import RegimeGrouping


def pooling_mask(grouping: RegimeGrouping, K: int) -> Tensor:
    """[B, R * K, Nc] bool, True = memory query may read the row. Queries of
    slot r read the rows of slot r; queries of an absent slot read every row
    (the empty-slot guard: no all-False mask row)."""
    present = grouping.slot_present[..., None]  # [B, R, 1]
    mask = grouping.slot_member | ~present  # [B, R, Nc]
    return mask.repeat_interleave(K, dim=1)


class RegimeMemoryPool(nn.Module):
    """Pool every regime of every variable into K memory tokens.

    K learned queries `q_mem` (one set, shared across regimes and variables)
    are repeated for each of the R slots; the queries of slot r attend only to
    the rows of slot r. Output = q_mem + attention, then FFN (pre-norm, as in
    MaskedSetAttentionLayer).

    Empty-slot guard: a slot padded up to R_max has no rows, so its mask row
    would be all False and softmax would give NaN (which also poisons the
    backward pass, even if the value is discarded; torch >= 2.5 SDPA returns
    zeros instead, but this does not rely on it). Absent slots are therefore
    allowed to see every row, and their memories are multiplied by
    slot_present afterwards; downstream attention masks them out as keys.

    Count feature (count_feature=True): PMA is a softmax average and cannot see
    the size of a regime, so Linear(1, d)(log1p(n_r)) is added to every memory
    of slot r.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_memory_tokens: int,
        count_feature: bool,
    ):
        super().__init__()
        if num_memory_tokens < 1:
            raise ValueError("reg_axis_num_memory_tokens must be >= 1")
        self.K = num_memory_tokens
        self.q_mem = nn.Parameter(torch.randn(num_memory_tokens, d_model))
        self.attn = MaskedSetAttentionLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.count = nn.Linear(1, d_model) if count_feature else None

    def forward(self, h: Tensor, grouping: RegimeGrouping) -> Tensor:
        """
        Args:
            h: [B, D, Nc, d] context tokens as per-variable sets.
            grouping: regime structure of the batch.

        Returns:
            memories [B, D, R, K, d], exactly zero on absent slots.
        """
        B, D, _, d = h.shape
        R, K = grouping.num_slots, self.K
        present = grouping.slot_present  # [B, R]
        q = self.q_mem.to(h.dtype).repeat(R, 1).expand(B, D, R * K, d)
        mem = self.attn(q, h, mask=pooling_mask(grouping, K)).view(B, D, R, K, d)
        if self.count is not None:
            n = torch.log1p(grouping.n_r.to(h.dtype))[..., None]  # [B, R, 1]
            mem = mem + self.count(n)[:, None, :, None, :]
        return mem * present[:, None, :, None, None].to(h.dtype)
