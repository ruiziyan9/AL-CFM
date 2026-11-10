"""Two-stage regime attention for reg_axis_full.

For one variable j and one reader token x (a context cell or a query cell),
with keys h[s, k] = the cell of variable j in row k of regime s:

    stage 1, for each source regime s (softmax over the rows k of s only):
        alpha[s, k] = softmax_k(<W_Q x, W_K h[s, k]> / sqrt(d / H))
        m[s]        = sum_k alpha[s, k] W_V h[s, k]
                      + count(log1p(n_s))                   (count feature)
                      + e_own * 1[s is the reader's regime] (context readers)

    stage 2 (softmax over the regimes present in the task):
        w[s] = softmax_s(<U_Q x, U_K norm(m[s])> / sqrt(d / H))
        x   <- x + W_O sum_s w[s] m[s]
        x   <- x + FFN(x)

Notation: B tasks (x query groups), L reader rows, Nc context rows,
D variables, d = d_model, H heads, R = R_max regimes, N = max regime size.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from CITNP.models.regime_axis_full.packing import RegimePacking, gather_keys


class TwoStageRegimeAttentionLayer(nn.Module):
    """Pre-norm two-stage regime attention + GELU FFN, both residual.

    Same conventions as `MaskedSetAttentionLayer`: bias-free q/k/v
    projections, output projection with dropout, one LayerNorm (`norm1`)
    shared by readers and keys. Stage 1 has H heads, concatenated to messages
    m[s] of width d. Stage 2 has H heads too; its value is the message itself
    (no value projection) and `regime_norm` is applied to the messages for
    scoring only.

    Args:
        d_model, nhead, dim_feedforward, dropout: as for the other layers.
        count_feature: add Linear(1, d)(log1p(n_s)) to the message of regime s.
        own_marker: build `e_own` [d] (zero-initialised), added to the message
            of the reader's own regime. For context readers only.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        count_feature: bool,
        own_marker: bool,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        # stage 1: reader -> rows of one regime
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.count = nn.Linear(1, d_model) if count_feature else None
        self.e_own = nn.Parameter(torch.zeros(d_model)) if own_marker else None
        # stage 2: reader -> regime messages
        self.regime_q = nn.Linear(d_model, d_model, bias=False)
        self.regime_k = nn.Linear(d_model, d_model, bias=False)
        self.regime_norm = nn.LayerNorm(d_model)
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

    def _pack_heads(self, keys: Tensor, index: Tensor) -> Tensor:
        """[B, Nc, D, d] -> packed keys [B, R, D * H, N, d / H]."""
        B, _, D, d = keys.shape
        _, R, N = index.shape
        H = self.nhead
        packed = gather_keys(keys, index)  # [B, R, N, D, d]
        packed = packed.view(B, R, N, D, H, d // H).permute(0, 1, 3, 4, 2, 5)
        return packed.reshape(B, R, D * H, N, d // H)

    def messages(
        self,
        readers: Tensor,
        context: Tensor,
        packing: RegimePacking,
        own: Optional[Tensor] = None,
    ) -> Tensor:
        """Stage 1: one message per (reader cell, source regime).

        Args:
            readers: [B, L, D, d] normalised reader tokens.
            context: [B, Nc, D, d] normalised context tokens (the keys).
            packing: regime packing of the context.
            own: [B, L, R] bool, the reader's own regime; needs `own_marker`.

        Returns:
            [B, D, L, R, d]. The message of an absent regime is finite but
            meaningless; stage 2 never attends to it.
        """
        B, L, D, d = readers.shape
        H, R = self.nhead, packing.num_slots
        # Readers stay flat and broadcast over the regime axis.
        q = self.to_q(readers).view(B, L, D, H, d // H).permute(0, 2, 3, 1, 4)
        q = q.reshape(B, 1, D * H, L, d // H)
        k = self._pack_heads(self.to_k(context), packing.index)
        v = self._pack_heads(self.to_v(context), packing.index)
        # [B, R, 1, 1, N], broadcast over variables, heads and readers.
        mask = packing.key_mask[:, :, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # [B, R, D * H, L, d / H] -> [B, D, L, R, d]
        out = out.view(B, R, D, H, L, d // H).permute(0, 2, 4, 1, 3, 5)
        msg = out.reshape(B, D, L, R, d)
        if self.count is not None:
            log_n = torch.log1p(packing.n_r.to(msg.dtype))[..., None]  # [B, R, 1]
            msg = msg + self.count(log_n)[:, None, None]
        if own is not None:
            if self.e_own is None:
                raise ValueError("own regimes given to a layer without e_own")
            msg = msg + own[:, None, :, :, None].to(msg.dtype) * self.e_own
        return msg

    def combine(self, readers: Tensor, msg: Tensor, present: Tensor) -> Tensor:
        """Stage 2: each reader cell attends over its regime messages.

        Args:
            readers: [B, L, D, d] normalised reader tokens.
            msg: [B, D, L, R, d] stage-1 messages.
            present: [B, R] bool, regimes that exist (at least one per task).

        Returns:
            [B, L, D, d] update, after the output projection.
        """
        B, L, D, d = readers.shape
        H, R = self.nhead, msg.shape[3]
        q = self.regime_q(readers).view(B, L, D, H, 1, d // H).transpose(1, 2)
        k = self.regime_k(self.regime_norm(msg))
        k = k.view(B, D, L, R, H, d // H).transpose(3, 4)
        v = msg.view(B, D, L, R, H, d // H).transpose(3, 4)
        # q [B, D, L, H, 1, d / H]; k, v [B, D, L, H, R, d / H]
        mask = present[:, None, None, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.reshape(B, D, L, d).transpose(1, 2)
        return self.to_out(out)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        packing: RegimePacking,
        own: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, L, D, d] reader tokens (context or query cells).
            context: [B, Nc, D, d] context tokens; the keys of variable j are
                the context cells of variable j.
            packing: regime packing of the context (`pack_regimes`).
            own: optional [B, L, R] bool, the own regime of each reader row
                (context readers; `packing.own`). None for query readers.

        Returns:
            [B, L, D, d].
        """
        readers = self.norm1(x)
        keys = readers if context is x else self.norm1(context)
        msg = self.messages(readers, keys, packing, own)
        update = self.combine(readers, msg, packing.slot_present)
        x = x + self.attn_dropout(update)
        return x + self.ff_block(self.norm2(x))
