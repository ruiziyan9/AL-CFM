"""
Mixed-context MACE-TNP (design R1).

The model predicts p(X_Y | do(X_T = t), C) where the context C mixes
observational rows and rows from interventional experiments. Each question
(T, Y) is exactly one upstream MACE-TNP question: the trainer flattens a task's
question groups into the batch dimension and calls this model with per-element
`intervention_index` / `outcome_index`, exactly like upstream.

The only addition to upstream is an intervened-flag embedding on context cells
that were set by an experiment (`context_int_mask`). Its last layer is
zero-initialised, so with an all-False mask, or before training, the model is
exactly upstream MACE-TNP and upstream checkpoints load directly
(`load_upstream_checkpoint`).
"""

from os import PathLike
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.causalinferencemodel import (
    CausalInferenceModel,
    MoGCausalInferenceModel,
)
from CITNP.models.causaltransformercomponents import CausalInfEncoder
from CITNP.models.nncomponents import build_mlp
from CITNP.utils.outputs import ModelOutput

# Parameters that upstream checkpoints do not have.
NEW_PARAM_PREFIX = "encoder.context_intervened."


class MixedCausalInfEncoder(CausalInfEncoder):
    """CausalInfEncoder plus an embedding for intervened context cells.

    `context_intervened` is added to the role-based context embedding on every
    cell with context_int_mask True. Its last Linear layer starts at zero, so
    the encoder starts out identical to upstream.
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        num_layers,
        num_nodes,
        device,
        dtype,
        emb_depth: int = 1,
        dropout: Optional[float] = 0.0,
        sample_attn_mode: str = "MHSA",
        linear_attention: bool = False,
    ):
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
            sample_attn_mode=sample_attn_mode,
            linear_attention=linear_attention,
        )
        self.context_intervened = build_mlp(
            dim_in=1, dim_hid=d_model, dim_out=d_model, depth=emb_depth
        )
        last = [
            m
            for m in self.context_intervened.modules()
            if isinstance(m, nn.Linear)
        ][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def encode(
        self,
        context_data: Tensor,
        target_data: Tensor,
        outcome_indices: Tensor,
        intervention_indices: Tensor,
        target_train_data: Optional[Tensor] = None,
        variable_mask: Optional[Tensor] = None,
        context_int_mask: Optional[Tensor] = None,
    ):
        """
        Upstream CausalInfEncoder.encode, verbatim except for the context
        embedding, which adds `context_intervened(x)` on intervened cells.

        Args:
        -----
        - context_int_mask (torch.Tensor, optional): bool [batch_size,
            num_context, num_nodes], True where the context cell was set by an
            intervention. None behaves like an all-False mask (upstream).
        """
        context_embed = self.embed(
            data=context_data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
            context=True,
        )
        if context_int_mask is not None:
            # [B, Nc, D, 1]
            flag = context_int_mask.unsqueeze(-1).to(context_embed.dtype)
            context_embed = context_embed + flag * self.context_intervened(
                context_data
            )
        target_embed = self.embed(
            data=target_data,
            outcome_indices=outcome_indices,
            intervention_indices=intervention_indices,
            context=False,
        )
        if target_train_data is not None:
            target_train_embed = self.embed(
                data=target_train_data,
                outcome_indices=outcome_indices,
                intervention_indices=intervention_indices,
                context=False,
            )
        else:
            target_train_embed = None

        # shape [batch_size, num_context + num_target, num_nodes, d_model]
        full_data = self.create_input(
            context_embed, target_embed, target_train_embed
        )
        sample_mask, num_target = self.create_sample_mask(
            context_embed, target_embed, target_train_embed
        )
        # shape [batch_size, num_samples, num_nodes, d_model]
        # Encode the data
        # shape [batch_size, num_samples, num_nodes, d_model]
        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            num_target=num_target,
            variable_mask=variable_mask,
        )

        # Compute the summary representation
        # shape [batch_size, num_target, num_nodes, d_model]
        return representation, num_target


# The mixin calls upstream model methods; give type checkers that base only.
if TYPE_CHECKING:
    _MixinBase = CausalInferenceModel
else:
    _MixinBase = object


class _MixedContextMixin(_MixinBase):
    """Swaps in MixedCausalInfEncoder and passes context_int_mask to it.

    Must come first in the bases, so its __init__ / forward wrap the upstream
    model's (predictor and loss come from the upstream class).
    """

    def __init__(self, config):
        super().__init__(config)  # type: ignore[call-arg]
        self.encoder = MixedCausalInfEncoder(
            d_model=self.d_model,
            emb_depth=self.emb_depth,
            dim_feedforward=self.dim_feedforward,
            nhead=self.nhead,
            dropout=self.dropout,
            num_layers=self.num_layers_encoder,
            num_nodes=self.num_nodes,
            device=self.device,
            dtype=self.dtype,
            sample_attn_mode=self.sample_attn_mode,
            linear_attention=self.linear_attention,
        )

    def forward(  # type: ignore[override]  # adds context_int_mask before variable_mask
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        outcome_index: torch.Tensor,
        context_int_mask: Optional[torch.Tensor] = None,
        variable_mask: Optional[torch.Tensor] = None,
    ):
        """CausalInferenceModel.forward, plus context_int_mask for the encoder.

        Shapes as upstream: context_data [B, Nc, D, 1], target_data
        [B, Nq, D, 1], intervention_index / outcome_index [B] (long),
        context_int_mask [B, Nc, D] (bool). B is the flattened (task, group)
        batch; the model itself knows nothing about groups.
        """
        masked_target_data = self._mask_target_data(
            target_data=target_data, intervention_index=intervention_index
        )
        # shape rep: (batch_size, num_target, num_nodes, d_model)
        rep, num_trgt = self.encoder.encode(
            context_data=context_data,
            target_data=masked_target_data,
            target_train_data=None,
            outcome_indices=outcome_index,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
            context_int_mask=context_int_mask,
        )
        # shape outcome_encoding: (batch_size, num_target, d_model)
        outcome_encoding, _ = self._extract_outcome_representation(
            representation=rep,
            outcome_index=outcome_index,
            num_trgt=num_trgt,
        )
        if self.predictor.output_heads == ("mean", "std"):
            mean, std = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std)
        elif self.predictor.output_heads == ("mean", "std", "weights"):
            mean, std, weights = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std, weights=weights)


class MixedContextModel(_MixedContextMixin, CausalInferenceModel):
    """Gaussian-predictive mixed-context model (CNP loss)."""


class MixedContextMoGModel(_MixedContextMixin, MoGCausalInferenceModel):
    """Mixture-of-Gaussians mixed-context model (MoG predictor and loss)."""


def load_upstream_checkpoint(
    model: nn.Module, path_or_state_dict: str | PathLike | dict
):
    """Load upstream MACE-TNP weights into a mixed-context model.

    Accepts a checkpoint path, a CheckpointCallback dict (state dict under
    "model_state_dict"), or a raw state dict. The only parameters allowed to be
    missing are the new `encoder.context_intervened.*` ones, which keep their
    zero-initialised last layer; unexpected keys are not allowed.

    Returns the `load_state_dict` result.
    """
    if isinstance(path_or_state_dict, dict):
        state = path_or_state_dict
    else:
        state = torch.load(path_or_state_dict, map_location="cpu")
    sd = state.get("model_state_dict", state)
    result = model.load_state_dict(sd, strict=False)
    expected = {k for k in model.state_dict() if k.startswith(NEW_PARAM_PREFIX)}
    if set(result.missing_keys) != expected:
        raise RuntimeError(
            "Upstream checkpoint mismatch: missing keys "
            f"{sorted(set(result.missing_keys) ^ expected)} differ from the "
            "expected encoder.context_intervened.* parameters"
        )
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in upstream checkpoint: {result.unexpected_keys}"
        )
    return result
