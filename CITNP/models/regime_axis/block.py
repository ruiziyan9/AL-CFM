"""One reg_axis block.

Notation: B tasks (x query groups), Nc context rows, Nq query rows,
D variables, d = d_model, K memory tokens, R = R_max regime slots.
Context tokens hc [B, Nc, D, d]; query tokens hq [B, Nq, D, d].

    0   hc += inject_var(code)                      (flag, optional)
    1   variable attention, context and query rows, shared weights
    0'  hc += inject_sample(code)                   (flag, optional)
    2   within-regime sample attention per variable (mask same_regime)
    3   PMA: K memories per (regime, variable), + count feature
    4   regime attention per variable over the R * K memories
    5   context feedback: context cells read their variable's memories
        (not built in the last block: it could never reach a prediction)
    6   query read: query cells read their variable's memories
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.regime_axis.attention import (
    MaskedSetAttentionLayer,
    from_flat_variable_sets,
    from_variable_sets,
    to_flat_variable_sets,
    to_row_stack,
    to_variable_sets,
)
from CITNP.models.regime_axis.grouping import RegimeGrouping
from CITNP.models.regime_axis.pooling import RegimeMemoryPool
from CITNP.networks.attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)


def _memory_mask(grouping: RegimeGrouping, K: int, D: int, L: int) -> Tensor:
    """Key-padding mask [B * D, L, R * K] (True = attend) over the memories."""
    keys = grouping.slot_present.repeat_interleave(K, dim=1)  # [B, R * K]
    B, RK = keys.shape
    return keys[:, None, None, :].expand(B, D, L, RK).reshape(B * D, L, RK)


class RegimeAxisBlock(nn.Module):
    """Steps 0-6 above. Steps 1, 4, 5, 6 reuse the upstream attention layers
    (small masks); steps 2 and 3 use MaskedSetAttentionLayer (broadcast
    [B, Nc, Nc] / [B, R * K, Nc] masks)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_memory_tokens: int,
        count_feature: bool,
        feedback: bool,
    ):
        super().__init__()
        up = dict(
            embed_dim=d_model,
            num_heads=nhead,
            head_dim=d_model // nhead,
            feedforward_dim=dim_feedforward,
            p_dropout=dropout,
            norm_first=True,
            activation=nn.GELU(),
            linear=False,
        )
        self.K = num_memory_tokens
        self.var_attn = MultiHeadSelfAttentionLayer(**up)
        self.sample_attn = MaskedSetAttentionLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.pool = RegimeMemoryPool(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            num_memory_tokens,
            count_feature,
        )
        self.regime_attn = MultiHeadSelfAttentionLayer(**up)
        self.feedback = MultiHeadCrossAttentionLayer(**up) if feedback else None
        self.query_read = MultiHeadCrossAttentionLayer(**up)

    def variable_attention(self, hc: Tensor, hq: Tensor) -> tuple[Tensor, Tensor]:
        """Step 1: attention across the D variables of each row (shared weights
        for context and query rows; rows never mix)."""
        B, Nc, D, d = hc.shape
        rows = to_row_stack(torch.cat([hc, hq], dim=1))  # [B * (Nc + Nq), D, d]
        out = self.var_attn(rows, mask=None).view(B, -1, D, d)
        return out[:, :Nc], out[:, Nc:]

    def sample_attention(self, hc: Tensor, grouping: RegimeGrouping) -> Tensor:
        """Step 2: [B, Nc, D, d] -> same; rows attend only within their regime."""
        x = to_variable_sets(hc)  # [B, D, Nc, d]
        return from_variable_sets(self.sample_attn(x, mask=grouping.same_regime))

    def memories(self, hc: Tensor, grouping: RegimeGrouping) -> Tensor:
        """Step 3: [B, Nc, D, d] -> memories [B, D, R, K, d]."""
        return self.pool(to_variable_sets(hc), grouping)

    def regime_attention(self, mem: Tensor, grouping: RegimeGrouping) -> Tensor:
        """Step 4: memories of all regimes of one variable attend to each
        other; absent slots are masked as keys and zeroed. -> [B * D, R * K, d].
        """
        B, D, R, K, d = mem.shape
        flat = mem.reshape(B * D, R * K, d)
        out = self.regime_attn(flat, mask=_memory_mask(grouping, K, D, R * K))
        keep = grouping.slot_present.repeat_interleave(K, dim=1)  # [B, R * K]
        keep = (
            keep[:, None, :, None].expand(B, D, R * K, 1).reshape(B * D, R * K, 1)
        )
        return out * keep.to(out.dtype)

    def read(
        self, layer: nn.Module, h: Tensor, mem: Tensor, grouping: RegimeGrouping
    ) -> Tensor:
        """Steps 5/6: cells of variable j ([B, N, D, d]) cross-attend to the
        fused memories of variable j ([B * D, R * K, d])."""
        B, N, D, _ = h.shape
        mask = _memory_mask(grouping, self.K, D, N)
        out = layer(to_flat_variable_sets(h), mem, mask=mask)
        return from_flat_variable_sets(out, B)

    def forward(
        self,
        hc: Tensor,
        hq: Tensor,
        grouping: RegimeGrouping,
        inject_var: Optional[Tensor] = None,
        inject_sample: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            hc: [B, Nc, D, d] context tokens; hq: [B, Nq, D, d] query tokens.
            grouping: regime structure of the context.
            inject_var / inject_sample: optional [B, Nc, D, d] flag additions
                before steps 1 and 2.

        Returns:
            (hc, hq) with the same shapes; hc is returned unchanged after step 2
            in a block without feedback (it is not used afterwards).
        """
        if inject_var is not None:
            hc = hc + inject_var
        hc, hq = self.variable_attention(hc, hq)
        if inject_sample is not None:
            hc = hc + inject_sample
        hc = self.sample_attention(hc, grouping)
        mem = self.regime_attention(self.memories(hc, grouping), grouping)
        if self.feedback is not None:
            hc = self.read(self.feedback, hc, mem, grouping)
        hq = self.read(self.query_read, hq, mem, grouping)
        return hc, hq
