"""reg_axis model classes.

The upstream model supplies the target masking, outcome extraction,
predictor and loss; `_RegimeAxisMixin` swaps in RegimeAxisEncoder and
forwards `context_int_mask`, `context_scale` and (optionally)
`context_regime` to it. The forward pass is `_encode_query_tokens` followed
by the prediction head (`MarginalPredictionHeadMixin`, or
`JointPredictionHeadMixin` for the joint model). This module must not import
CITNP.models.mixed_context_model (which imports this package).
"""

from typing import Optional

from torch import Tensor

from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    MoGCausalInferenceModel,
)
from CITNP.models.prediction_heads import (
    JointPredictionHeadMixin,
    MarginalPredictionHeadMixin,
)
from CITNP.models.regime_axis.encoder import (
    JointRegimeAxisEncoder,
    RegimeAxisEncoder,
)
from CITNP.models.scale_conditioning import check_scale_conditioning

# Parameters that only reg_axis models have (no upstream counterpart).
REG_AXIS_PARAM_PREFIXES = ("encoder.blocks.", "encoder.flag.")
# CausalInfModelConfig fields read by reg_axis (all have defaults).
REG_AXIS_CONFIG_KEYS = (
    "reg_axis_num_blocks",
    "reg_axis_num_memory_tokens",
    "reg_axis_target_flag",
    "reg_axis_count_feature",
)


class _RegimeAxisMixin(MarginalPredictionHeadMixin):
    """Swaps in RegimeAxisEncoder; forward matches _MixedContextMixin.forward.

    Must come first in the bases (its __init__ / forward wrap the upstream
    model's). `sample_attn_mode` is ignored; `linear_attention` must be False.
    """

    encoder_cls: type[RegimeAxisEncoder] = RegimeAxisEncoder

    def __init__(self, config):
        super().__init__(config)  # type: ignore[call-arg]
        self.scale_conditioning = check_scale_conditioning(
            getattr(config, "scale_conditioning", "none")
        )
        self.encoder = self.encoder_cls(
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
            num_memory_tokens=getattr(config, "reg_axis_num_memory_tokens", 4),
            target_flag=getattr(config, "reg_axis_target_flag", "injected"),
            count_feature=getattr(config, "reg_axis_count_feature", True),
        )

    def forward(  # type: ignore[override]  # mixed-context signature
        self,
        context_data: Tensor,
        target_data: Tensor,
        intervention_index: Tensor,
        outcome_index: Optional[Tensor] = None,
        context_int_mask: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
        context_scale: Optional[Tensor] = None,
        context_regime: Optional[Tensor] = None,
    ):
        """
        Args:
            context_data: [B, Nc, D, 1]; target_data: [B, Nq, D, 1].
            intervention_index / outcome_index: [B] long (T / Y); outcome_index
                may be None only for the joint model (no outcome role).
            context_int_mask: [B, Nc, D] bool; None = all observational.
            variable_mask: must be None (not supported).
            context_scale: [B, D, 2] (log std, mean / std); required iff
                scale_conditioning is "stats", ignored otherwise.
            context_regime: [B, Nc] long, consistency check only.

        Returns:
            ModelOutput with pred_mean / pred_std (and weights for MoG),
            each [B, Nq, ...] as for the other mixed-context models; a
            JointModelOutput for the joint model.
        """
        query_tokens = self._encode_query_tokens(
            context_data=context_data,
            target_data=target_data,
            intervention_index=intervention_index,
            outcome_index=outcome_index,
            context_int_mask=context_int_mask,
            variable_mask=variable_mask,
            context_scale=context_scale,
            context_regime=context_regime,
        )
        return self._predict_from_query_tokens(
            query_tokens, intervention_index, outcome_index
        )

    def _encode_query_tokens(
        self,
        context_data: Tensor,
        target_data: Tensor,
        intervention_index: Tensor,
        outcome_index: Optional[Tensor],
        context_int_mask: Optional[Tensor],
        variable_mask: Optional[Tensor],
        context_scale: Optional[Tensor],
        context_regime: Optional[Tensor],
    ) -> Tensor:
        """The encoder part of `forward`: query tokens [B, Nq, D, d_model].
        Arguments as for `forward`; without outcome_index the encoder gets
        intervention_index as its outcome_indices (only outcome-free encoders
        are called that way)."""
        if variable_mask is not None:
            raise ValueError("reg_axis does not support variable_mask")
        if self.scale_conditioning == "stats" and context_scale is None:
            raise ValueError("scale_conditioning=stats needs context_scale")
        masked = self._mask_target_data(
            target_data=target_data, intervention_index=intervention_index
        )
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked,
            outcome_indices=(
                intervention_index if outcome_index is None else outcome_index
            ),
            intervention_indices=intervention_index,
            context_int_mask=context_int_mask,
            context_scale=(
                context_scale if self.scale_conditioning == "stats" else None
            ),
            context_regime=context_regime,
        )
        return rep[:, -num_trgt:]


class RegimeAxisMixedContextModel(_RegimeAxisMixin, CausalInferenceModel):
    """Gaussian-predictive model with the regime-axis encoding (CNP loss)."""


class RegimeAxisMixedContextMoGModel(_RegimeAxisMixin, MoGCausalInferenceModel):
    """Mixture-of-Gaussians model with the regime-axis encoding."""


class JointRegimeAxisMixedContextMoGModel(
    JointPredictionHeadMixin, RegimeAxisMixedContextMoGModel
):
    """The reg_axis encoding with the joint prediction head and no outcome
    role (CITNP/models/prediction_heads.py). Set
    joint_head_node_attention_layers to 1: every reg_axis block ends with a
    per-variable read of the context, so the variable tokens reaching the
    head have not been mixed.

    Each regime is compressed per variable into memory tokens, so dependence
    between variables inside a regime reaches the query only through the
    variable attention that precedes the compression. Expect this encoding to
    be the weakest for joint prediction (an empirical question)."""

    encoder_cls = JointRegimeAxisEncoder
