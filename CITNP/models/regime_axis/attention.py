"""Masked set attention for reg_axis, plus reshape helpers.

Upstream attention layers copy a 2-D/3-D mask once per head
(`einops.repeat`), which for the within-regime sample attention would cost
B * D * H * Nc**2 booleans. `MaskedSetAttentionLayer` instead keeps one
[B, L, S] mask and lets SDPA broadcast it over variables and heads.

Notation: B tasks, N rows, D variables, d = d_model, H heads.
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def to_variable_sets(h: Tensor) -> Tensor:
    """[B, N, D, d] -> [B, D, N, d]: one set of rows per (task, variable)."""
    return h.permute(0, 2, 1, 3)


def from_variable_sets(h: Tensor) -> Tensor:
    """[B, D, N, d] -> [B, N, D, d] (inverse of to_variable_sets)."""
    return h.permute(0, 2, 1, 3)


def to_row_stack(h: Tensor) -> Tensor:
    """[B, N, D, d] -> [B * N, D, d]: one token set per row."""
    B, N, D, d = h.shape
    return h.reshape(B * N, D, d)


def to_flat_variable_sets(h: Tensor) -> Tensor:
    """[B, N, D, d] -> [B * D, N, d]: per-variable sets for upstream layers."""
    B, N, D, d = h.shape
    return to_variable_sets(h).reshape(B * D, N, d)


def from_flat_variable_sets(h: Tensor, B: int) -> Tensor:
    """[B * D, N, d] -> [B, N, D, d] (inverse of to_flat_variable_sets)."""
    BD, N, d = h.shape
    return from_variable_sets(h.view(B, BD // B, N, d))


class MaskedSetAttentionLayer(nn.Module):
    """Pre-norm multi-head attention + GELU FFN, both residual.

    Same structure as upstream MultiHeadSelfAttentionLayer /
    MultiHeadCrossAttentionLayer with norm_first=True (bias-free q/k/v
    projections, output projection with dropout, one LayerNorm shared by
    queries and keys), but the boolean mask [B, L, S] (True = attend, the
    SDPA convention) is broadcast over variables and heads instead of copied.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(dropout)
        )
        self.ff_block = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:
        """[B, D, L, d] -> [B, D * H, L, d / H] (variables and heads merged,
        so SDPA sees 4-D inputs and a [B, 1, L, S] mask broadcasts)."""
        B, D, L, d = x.shape
        x = x.view(B, D, L, self.nhead, d // self.nhead).permute(0, 1, 3, 2, 4)
        return x.reshape(B, D * self.nhead, L, d // self.nhead)

    def _attend(self, xq: Tensor, xkv: Tensor, mask: Optional[Tensor]) -> Tensor:
        B, D, L, d = xq.shape
        q = self._split(self.to_q(xq))
        k = self._split(self.to_k(xkv))
        v = self._split(self.to_v(xkv))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask[:, None]
        )  # [B, D * H, L, d / H]
        out = out.view(B, D, self.nhead, L, d // self.nhead).permute(
            0, 1, 3, 2, 4
        )
        return self.to_out(out.reshape(B, D, L, d))

    def forward(
        self,
        xq: Tensor,
        xkv: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            xq: [B, D, L, d] queries (and keys/values if xkv is None).
            xkv: optional [B, D, S, d] keys/values (cross-attention).
            mask: optional bool [B, L, S], True = attend; every query row needs
                at least one True entry (else SDPA returns NaN).

        Returns:
            [B, D, L, d].
        """
        q_in = self.norm1(xq)
        kv_in = q_in if xkv is None else self.norm1(xkv)
        x = xq + self.attn_dropout(self._attend(q_in, kv_in, mask))
        return x + self.ff_block(self.norm2(x))
