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

`LayerwiseMixedContext{,MoG}Model` additionally re-injects the flag signal
before every attention block of the encoder (see
`LayerwiseMixedCausalInfEncoder`). Its new layers are zero-initialised too, so
it starts out identical to the R1 model and R1 checkpoints load directly
(`load_r1_checkpoint`).
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
from CITNP.models.causaltransformercomponents import (
    CausalInfEncoder,
    CausalInfTransformerEncoder,
)
from CITNP.models.nncomponents import build_mlp
from CITNP.utils.outputs import ModelOutput
from CITNP.utils.utils import (
    reshape_back_node_attention,
    reshape_back_sample_attention,
    reshape_for_node_attention,
    reshape_for_sample_attention,
)

# Parameters that upstream checkpoints do not have.
NEW_PARAM_PREFIX = "encoder.context_intervened."
# Parameters that R1 (embedding-only flag) checkpoints do not have.
LAYERWISE_PARAM_PREFIXES = ("encoder.flag_code.", "encoder.encoder.inject_")


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
            **self._extra_encoder_inputs(
                context_data, context_int_mask, full_data
            ),
        )

        # Compute the summary representation
        # shape [batch_size, num_target, num_nodes, d_model]
        return representation, num_target

    def _extra_encoder_inputs(
        self,
        context_data: Tensor,
        context_int_mask: Optional[Tensor],
        full_data: Tensor,
    ) -> dict:
        """Extra keyword inputs for the transformer stack; none for R1."""
        return {}


class InjectedCausalInfTransformerEncoder(CausalInfTransformerEncoder):
    """Upstream transformer stack plus layer-wise re-injection of the flag.

    Before every sample-attention block and every node-attention block, the
    flag code (zero on non-intervened cells and on target rows) is added to
    the residual stream through its own Linear map:

        src <- src + inject_sample[l](flag_code)   (before sample attention)
        src <- src + inject_node[l](flag_code)     (before node attention)

    There is no node injection in the last block: after the last sample
    attention, context rows no longer reach the target rows (node attention
    mixes within a row only), so it could never affect the prediction.
    The maps have no bias, so non-intervened cells are never touched, and
    zero-initialised weights, so the stack starts out identical to upstream.
    forward() is upstream's verbatim apart from the two marked additions.
    """

    def __init__(
        self,
        sample_layers: nn.ModuleList,
        node_layers: nn.ModuleList,
        d_model: int,
        sample_attn_mode: str = "MHSA",
    ) -> None:
        super().__init__(
            sample_layers=sample_layers,
            node_layers=node_layers,
            sample_attn_mode=sample_attn_mode,
        )
        n_blocks = len(self.iterator)
        self.inject_sample = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_blocks)
        )
        self.inject_node = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_blocks - 1)
        )
        for lin in (*self.inject_sample, *self.inject_node):
            nn.init.zeros_(lin.weight)

    def forward(  # type: ignore[override]  # adds flag_code
        self,
        src: Tensor,
        sample_mask: Tensor,
        num_target: int,
        variable_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        flag_code: Optional[Tensor] = None,
    ) -> Tensor:
        """flag_code: [batch_size, num_samples, num_nodes, d_model], zero on
        every row/cell that is not an intervened context cell. None means
        no injection (upstream behaviour)."""
        batch_size, num_samples, num_nodes, d_model = src.size()

        for i, (sample_layer, node_layer) in enumerate(self.iterator):
            # --- addition 1: re-inject before sample attention ---
            if flag_code is not None:
                src = src + self.inject_sample[i](flag_code)

            src = reshape_for_sample_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
            if self.sample_attn_mode == "MHSA":
                src = sample_layer(src, mask=sample_mask)
            elif self.sample_attn_mode == "MHCA":
                ctxt_src = src[:, :-num_target, :]
                tgt_src = src[:, -num_target:, :]
                ctxt_src = sample_layer[0](ctxt_src)
                tgt_src = sample_layer[1](tgt_src, ctxt_src)
                src = torch.cat([ctxt_src, tgt_src], dim=1)
            src = reshape_back_sample_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )

            # --- addition 2: re-inject before node attention ---
            if flag_code is not None and i < len(self.inject_node):
                src = src + self.inject_node[i](flag_code)

            src = reshape_for_node_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
            src = node_layer(src, mask=None)
            src = reshape_back_node_attention(
                src, batch_size, num_samples, num_nodes, d_model
            )
        return src


class LayerwiseMixedCausalInfEncoder(MixedCausalInfEncoder):
    """MixedCausalInfEncoder whose flag signal is re-injected at every block.

    Keeps R1's embedding-level `context_intervened` and adds
    - `flag_code`: an MLP of the context value, applied on intervened cells
      only (default init, so the injection maps get gradient from step 0);
    - the per-block injection maps inside `self.encoder`
      (InjectedCausalInfTransformerEncoder, zero-initialised).
    The upstream attention layers are reused as they are, so their parameter
    names, and therefore R1/upstream checkpoints, are unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs["d_model"]
        self.flag_code = build_mlp(
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=kwargs.get("emb_depth", 1),
        )
        old = self.encoder
        self.encoder = InjectedCausalInfTransformerEncoder(
            sample_layers=old.sample_layers,
            node_layers=old.node_layers,
            d_model=d_model,
            sample_attn_mode=old.sample_attn_mode,
        )

    def _extra_encoder_inputs(self, context_data, context_int_mask, full_data):
        if context_int_mask is None:
            return {}
        # [B, Nc, D, 1] -> code [B, Nc, D, d_model], zero off intervened cells
        flag = context_int_mask.unsqueeze(-1).to(full_data.dtype)
        code = flag * self.flag_code(context_data)
        # target (and target-train) rows never carry a flag
        pad = code.new_zeros(
            code.shape[0], full_data.shape[1] - code.shape[1], *code.shape[2:]
        )
        return {"flag_code": torch.cat([code, pad], dim=1)}


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

    encoder_cls: type[CausalInfEncoder] = MixedCausalInfEncoder

    def __init__(self, config):
        super().__init__(config)  # type: ignore[call-arg]
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


class LayerwiseMixedContextModel(MixedContextModel):
    """MixedContextModel with the flag re-injected before every block."""

    encoder_cls = LayerwiseMixedCausalInfEncoder


class LayerwiseMixedContextMoGModel(MixedContextMoGModel):
    """MixedContextMoGModel with the flag re-injected before every block."""

    encoder_cls = LayerwiseMixedCausalInfEncoder

# ---------------------------------------------------------------------------
# Intervention-encoding switch. The single place that maps a method name to a
# model class: mixed_train.py and evaluation/model_api.py both resolve here.
# ---------------------------------------------------------------------------

DEFAULT_INTERVENTION_ENCODING = "r1"
INTERVENTION_ENCODINGS: dict[str, type[CausalInferenceModel]] = {
    "r1": MixedContextMoGModel,
    "layerwise": LayerwiseMixedContextMoGModel,
}


def resolve_model_class(
    intervention_encoding: Optional[str] = None,
    model_class: Optional[str] = None,
) -> tuple[type[CausalInferenceModel], str]:
    """Return (model class, encoding name).

    - Neither given: the default encoding (r1, the upstream-compatible model).
    - intervention_encoding only: its registered (mixture-of-Gaussians) class.
    - model_class only (legacy configs): that class from this module; the
      returned name is its encoding if registered, else the class name.
    - Both: allowed only if they agree.
    """
    if (
        intervention_encoding is not None
        and intervention_encoding not in INTERVENTION_ENCODINGS
    ):
        raise ValueError(
            f"Unknown intervention_encoding {intervention_encoding!r}; "
            f"choose from {sorted(INTERVENTION_ENCODINGS)}"
        )
    if model_class is None:
        name = intervention_encoding or DEFAULT_INTERVENTION_ENCODING
        return INTERVENTION_ENCODINGS[name], name

    cls = globals().get(model_class)
    if not (isinstance(cls, type) and issubclass(cls, CausalInferenceModel)):
        raise ValueError(
            f"Unknown model_class {model_class!r} (not a model class in {__name__})"
        )
    matches = [k for k, v in INTERVENTION_ENCODINGS.items() if v is cls]
    name = matches[0] if matches else cls.__name__
    if intervention_encoding is not None and intervention_encoding != name:
        raise ValueError(
            f"model_class={model_class!r} is encoding {name!r}, which "
            f"contradicts intervention_encoding={intervention_encoding!r}"
        )
    return cls, name

def _load_with_new_params(
    model: nn.Module, path_or_state_dict, new_prefixes: tuple[str, ...]
):
    """Load a checkpoint that lacks exactly the model's parameters whose
    names start with one of `new_prefixes`; anything else is an error."""
    if isinstance(path_or_state_dict, dict):
        state = path_or_state_dict
    else:
        state = torch.load(path_or_state_dict, map_location="cpu")
    sd = state.get("model_state_dict", state)
    result = model.load_state_dict(sd, strict=False)
    expected = {
        k for k in model.state_dict() if k.startswith(tuple(new_prefixes))
    }
    if set(result.missing_keys) != expected:
        raise RuntimeError(
            "Checkpoint mismatch: missing keys "
            f"{sorted(set(result.missing_keys) ^ expected)} differ from the "
            f"expected new parameters {new_prefixes}"
        )
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {result.unexpected_keys}"
        )
    return result


def load_upstream_checkpoint(
    model: nn.Module, path_or_state_dict: str | PathLike | dict
):
    """Load upstream MACE-TNP weights into a mixed-context model.

    Accepts a checkpoint path, a CheckpointCallback dict (state dict under
    "model_state_dict"), or a raw state dict. The only parameters allowed to be
    missing are the new mixed-context ones (`encoder.context_intervened.*`,
    plus the layer-wise ones for Layerwise* models), which keep their
    zero-initialised last layers; unexpected keys are not allowed.

    Returns the `load_state_dict` result.
    """
    return _load_with_new_params(
        model,
        path_or_state_dict,
        (NEW_PARAM_PREFIX, *LAYERWISE_PARAM_PREFIXES),
    )


def load_r1_checkpoint(
    model: nn.Module, path_or_state_dict: str | PathLike | dict
):
    """Warm-start a Layerwise* model from an R1 (MixedContext*) checkpoint.

    Only the layer-wise parameters (`encoder.flag_code.*`,
    `encoder.encoder.inject_*`) may be missing; the injection maps stay at
    zero, so the loaded model reproduces the R1 model exactly.
    """
    return _load_with_new_params(
        model, path_or_state_dict, LAYERWISE_PARAM_PREFIXES
    )
