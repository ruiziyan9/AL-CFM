"""RegimeAxisFullEncoder: upstream role embedding + flag + scale +
reg_axis_full blocks.

Notation: B tasks (x query groups), Nc context rows, Nq query rows,
D variables, d = d_model.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.causaltransformercomponents import CausalInfEncoder
from CITNP.models.regime_axis import group_regimes
from CITNP.models.regime_axis.encoding import FlagInjection
from CITNP.models.regime_axis_full.block import RegimeAxisFullBlock
from CITNP.models.regime_axis_full.packing import pack_regimes
from CITNP.models.scale_conditioning import (
    ScaleConditioning,
    check_scale_conditioning,
)


class RegimeAxisFullEncoder(CausalInfEncoder):
    """Keeps the six upstream role MLPs (`context_variable`, ...,
    `target_outcome`) and `embed()`; replaces the upstream transformer stack
    (`self.encoder`, deleted) by `self.blocks`, a list of RegimeAxisFullBlock.

    Per cell, before the first block:
        h0 = role embedding + flag embedding (intervened context cells only)
             + scale embedding of the variable column (scale_conditioning stats)

    The regime structure is derived here from `context_int_mask`
    (`group_regimes`, then `pack_regimes`, once per forward pass); context
    rows may arrive in any order.

    `sample_attn_mode` is ignored; `linear_attention=True` is rejected
    (the regime masks need softmax attention).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        num_nodes: int,
        device,
        dtype,
        emb_depth: int = 1,
        dropout: Optional[float] = 0.0,
        sample_attn_mode: str = "MHSA",
        linear_attention: bool = False,
        scale_conditioning: str = "none",
        num_blocks: Optional[int] = None,
        target_flag: str = "injected",
        count_feature: bool = True,
    ):
        if linear_attention:
            raise ValueError(
                "reg_axis_full does not support linear_attention=True"
            )
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
            emb_depth=emb_depth,
            dropout=dropout,
        )
        del self.encoder  # upstream transformer stack, unused
        n = num_layers // 2 if num_blocks is None else num_blocks
        if n < 1:
            raise ValueError(f"reg_axis_full needs at least one block (got {n})")
        self.blocks = nn.ModuleList(
            RegimeAxisFullBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout or 0.0,
                count_feature=count_feature,
            )
            for _ in range(n)
        )
        self.flag = FlagInjection(d_model, emb_depth, n, target_flag)
        self.scale_conditioning = check_scale_conditioning(scale_conditioning)
        if self.scale_conditioning == "stats":
            self.scale_embedding = ScaleConditioning(d_model, emb_depth)

    def encode(  # type: ignore[override]  # different inputs and outputs
        self,
        context_data: Tensor,
        target_data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        context_int_mask: Optional[Tensor] = None,
        context_scale: Optional[Tensor] = None,
        context_regime: Optional[Tensor] = None,
    ) -> tuple[Tensor, int]:
        """
        Args:
            context_data: [B, Nc, D, 1].
            target_data: [B, Nq, D, 1], already masked (only T visible).
            outcome_indices / intervention_indices: [B] long (Y / T).
            context_int_mask: [B, Nc, D] bool; None = all observational.
            context_scale: [B, D, 2], required iff scale_conditioning stats.
            context_regime: [B, Nc] long, consistency check only.

        Returns:
            (query tokens [B, Nq, D, d], Nq).
        """
        B, Nc, D, _ = context_data.shape
        if context_int_mask is None:
            context_int_mask = torch.zeros(
                B, Nc, D, dtype=torch.bool, device=context_data.device
            )
        packing = pack_regimes(group_regimes(context_int_mask, context_regime))
        roles = dict(
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
        )
        hc = self.embed(data=context_data, context=True, **roles)
        hq = self.embed(data=target_data, context=False, **roles)
        flag_emb = self.flag.embed(context_data, context_int_mask)
        if flag_emb is not None:
            hc = hc + flag_emb
        if self.scale_conditioning == "stats":
            if context_scale is None:
                raise ValueError("scale_conditioning=stats needs context_scale")
            hc = self.scale_embedding.add_to(hc, context_scale)
            hq = self.scale_embedding.add_to(hq, context_scale)
        code = self.flag.code(context_data, context_int_mask)
        for i, block in enumerate(self.blocks):
            hc, hq = block(
                hc,
                hq,
                packing,
                inject_var=self.flag.inject(code, i, "var"),
                inject_sample=self.flag.inject(code, i, "sample"),
            )
        return hq, hq.shape[1]
