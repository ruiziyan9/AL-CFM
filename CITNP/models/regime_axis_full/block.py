"""One reg_axis_full block.

Notation: B tasks (x query groups), Nc context rows, Nq query rows,
D variables, d = d_model, R = R_max regimes, N = max regime size.
Context tokens hc [B, Nc, D, d]; query tokens hq [B, Nq, D, d].

    0   hc += inject_var(code)                      (flag, optional)
    1   variable attention, context and query rows, shared weights
    0'  hc += inject_sample(code)                   (flag, optional)
    2   context read: every context cell runs the two-stage regime
        attention over the context rows (keys = hc after 0'), with e_own
    3   query read: every query cell runs the two-stage regime attention
        over the UPDATED context rows of step 2, without e_own

Query rows are never keys, never attend to each other, and are never read by
context rows.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.regime_axis.attention import to_row_stack
from CITNP.models.regime_axis_full.attention import TwoStageRegimeAttentionLayer
from CITNP.models.regime_axis_full.packing import RegimePacking
from CITNP.networks.attention_layers import MultiHeadSelfAttentionLayer


class RegimeAxisFullBlock(nn.Module):
    """Steps 0-3 above. Step 1 reuses the upstream self-attention layer;
    steps 2 and 3 are two TwoStageRegimeAttentionLayer instances with
    separate weights. Every block has both: the context update of step 2
    reaches the prediction through step 3 of the same block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        count_feature: bool,
    ):
        super().__init__()
        self.var_attn = MultiHeadSelfAttentionLayer(
            embed_dim=d_model,
            num_heads=nhead,
            head_dim=d_model // nhead,
            feedforward_dim=dim_feedforward,
            p_dropout=dropout,
            norm_first=True,
            activation=nn.GELU(),
            linear=False,
        )
        self.context_read = TwoStageRegimeAttentionLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            count_feature,
            own_marker=True,
        )
        self.query_read = TwoStageRegimeAttentionLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            count_feature,
            own_marker=False,
        )

    def variable_attention(self, hc: Tensor, hq: Tensor) -> tuple[Tensor, Tensor]:
        """Step 1: attention across the D variables of each row (shared weights
        for context and query rows; rows never mix)."""
        B, Nc, D, d = hc.shape
        rows = to_row_stack(torch.cat([hc, hq], dim=1))  # [B * (Nc + Nq), D, d]
        out = self.var_attn(rows, mask=None).view(B, -1, D, d)
        return out[:, :Nc], out[:, Nc:]

    def forward(
        self,
        hc: Tensor,
        hq: Tensor,
        packing: RegimePacking,
        inject_var: Optional[Tensor] = None,
        inject_sample: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            hc: [B, Nc, D, d] context tokens; hq: [B, Nq, D, d] query tokens.
            packing: regime packing of the context.
            inject_var / inject_sample: optional [B, Nc, D, d] flag additions
                before steps 1 and 2.

        Returns:
            (hc, hq) with the same shapes.
        """
        if inject_var is not None:
            hc = hc + inject_var
        hc, hq = self.variable_attention(hc, hq)
        if inject_sample is not None:
            hc = hc + inject_sample
        hc = self.context_read(hc, hc, packing, own=packing.own)
        hq = self.query_read(hq, hc, packing)
        return hc, hq
