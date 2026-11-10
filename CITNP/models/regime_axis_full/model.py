"""reg_axis_full model classes.

The upstream model supplies the target masking, outcome extraction,
predictor and loss; `_RegimeAxisFullMixin` swaps in RegimeAxisFullEncoder and
forwards `context_int_mask`, `context_scale` and (optionally)
`context_regime` to it. This module must not import
CITNP.models.mixed_context_model (which imports this package).
"""

from typing import TYPE_CHECKING, Optional

from torch import Tensor

from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    MoGCausalInferenceModel,
)
from CITNP.models.regime_axis_full.encoder import RegimeAxisFullEncoder
from CITNP.models.scale_conditioning import check_scale_conditioning
from CITNP.utils.outputs import ModelOutput

# Parameters that only reg_axis_full models have (no upstream counterpart).
REG_AXIS_FULL_PARAM_PREFIXES = ("encoder.blocks.", "encoder.flag.")
# CausalInfModelConfig fields read by reg_axis_full (all have defaults).
# `reg_axis_num_memory_tokens` is not among them: there are no memory tokens.
REG_AXIS_FULL_CONFIG_KEYS = (
    "reg_axis_num_blocks",
    "reg_axis_target_flag",
    "reg_axis_count_feature",
)

if TYPE_CHECKING:
    _MixinBase = CausalInferenceModel
else:
    _MixinBase = object


class _RegimeAxisFullMixin(_MixinBase):
    """Swaps in RegimeAxisFullEncoder; forward matches
    _MixedContextMixin.forward.

    Must come first in the bases (its __init__ / forward wrap the upstream
    model's). `sample_attn_mode` is ignored; `linear_attention` must be False.
    """

    def __init__(self, config):
        super().__init__(config)  # type: ignore[call-arg]
        self.scale_conditioning = check_scale_conditioning(
            getattr(config, "scale_conditioning", "none")
        )
        self.encoder = RegimeAxisFullEncoder(
            d_model=self.d_model,
            emb_depth=self.emb_depth,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            num_layers=self.num_layers_encoder,
            num_nodes=self.num_nodes,
            device=self.device,
            dtype=self.dtype,
            linear_attention=self.linear_attention,
            scale_conditioning=self.scale_conditioning,
            num_blocks=getattr(config, "reg_axis_num_blocks", None),
            target_flag=getattr(config, "reg_axis_target_flag", "injected"),
            count_feature=getattr(config, "reg_axis_count_feature", True),
        )

    def forward(  # type: ignore[override]  # mixed-context signature
        self,
        context_data: Tensor,
        target_data: Tensor,
        intervention_index: Tensor,
        outcome_index: Tensor,
        context_int_mask: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
        context_scale: Optional[Tensor] = None,
        context_regime: Optional[Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            context_data: [B, Nc, D, 1]; target_data: [B, Nq, D, 1].
            intervention_index / outcome_index: [B] long (T / Y).
            context_int_mask: [B, Nc, D] bool; None = all observational.
            variable_mask: must be None (not supported).
            context_scale: [B, D, 2] (log std, mean / std); required iff
                scale_conditioning is "stats", ignored otherwise.
            context_regime: [B, Nc] long, consistency check only.

        Returns:
            ModelOutput with pred_mean / pred_std (and weights for MoG),
            each [B, Nq, ...] as for the other mixed-context models.
        """
        if variable_mask is not None:
            raise ValueError("reg_axis_full does not support variable_mask")
        if self.scale_conditioning == "stats" and context_scale is None:
            raise ValueError("scale_conditioning=stats needs context_scale")
        masked = self._mask_target_data(
            target_data=target_data, intervention_index=intervention_index
        )
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            context_int_mask=context_int_mask,
            context_scale=(
                context_scale if self.scale_conditioning == "stats" else None
            ),
            context_regime=context_regime,
        )
        outcome_encoding, _ = self._extract_outcome_representation(
            representation=rep, outcome_index=outcome_index, num_trgt=num_trgt
        )
        if self.predictor.output_heads == ("mean", "std"):
            mean, std = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std)
        mean, std, weights = self._predict(outcome_encoding)
        return ModelOutput(pred_mean=mean, pred_std=std, weights=weights)


class RegimeAxisFullMixedContextModel(_RegimeAxisFullMixin, CausalInferenceModel):
    """Gaussian-predictive model with the full regime-axis encoding (CNP loss)."""


class RegimeAxisFullMixedContextMoGModel(
    _RegimeAxisFullMixin, MoGCausalInferenceModel
):
    """Mixture-of-Gaussians model with the full regime-axis encoding."""
