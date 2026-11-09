"""The only code that calls a trained model: loading it, getting its
predictive mixture for interventional queries on the raw data scale, and
reading its internal context representations (`context_states`).

Everything else derives what it needs from `predict_mixture` via
`mixture_logpdf`, `mixture_cdf` and `effective_components`, or from
`context_states`.
"""

from typing import Any, Callable

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.trainer.mixed_context_trainer import _normalisation_stats
from CITNP.utils.configs import CausalInfModelConfig

DEFAULT_MODEL_CLASS = "MixedContextMoGModel"


def load_model(
    ckpt: str, model_config: str, device: str, num_nodes: int = 3
) -> torch.nn.Module:
    """Build the model described by the YAML `model_config` for `num_nodes`
    variables, load the weights in `ckpt` and return it in eval mode.

    The class is resolved by resolve_model_class from the optional YAML keys intervention_encoding and/or model_class (default: r1, MixedContextMoGModel).
    """
    from CITNP.models.mixed_context_model import resolve_model_class
    from CITNP.models.regime_axis import REG_AXIS_CONFIG_KEYS

    with open(model_config) as f:
        m = yaml.safe_load(f)
    cfg = CausalInfModelConfig(
        d_model=m["d_model"],
        emb_depth=m["emb_depth"],
        dim_feedforward=m["dim_feedforward"],
        nhead=m["nhead"],
        dropout=0.0,
        num_layers_encoder=m["num_layers_encoder"],
        num_nodes=num_nodes,
        sample_attn_mode=m["sample_attn_mode"],
        linear_attention=m.get("linear_attention", False),
        mean_loss_across_samples=False,
        device=device,
        dtype="float32",
        num_mixture_components=m["num_mixture_components"],
        scale_conditioning=m.get("scale_conditioning", "none"),
        # reg_axis only; absent keys keep the dataclass defaults
        **{k: m[k] for k in REG_AXIS_CONFIG_KEYS if k in m},
        # layerwise only; absent (older runs) = separate flag_code MLP
        share_intervention_encoder=bool(
            m.get("share_intervention_encoder", False)
        ),
    )
    model_cls, _ = resolve_model_class(
        m.get("intervention_encoding"), m.get("model_class")
    )
    model = model_cls(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    return model.to(device).eval()


def _model_inputs(ctx_x, ctx_int, T, Y, t, *, zero_flags, normalise):
    """Model keyword inputs (CPU, float32) for one context and Q queries,
    plus the per-variable standardisation (mean, std) [D] that was applied.
    `context_scale` [Q, D, 2] is (log std, mean / std) per variable (zeros
    without normalisation), used by models with scale_conditioning "stats"."""
    Ts = torch.as_tensor(np.atleast_1d(T), dtype=torch.long)
    Ys = torch.as_tensor(np.atleast_1d(Y), dtype=torch.long)
    if Ts.ndim != 1 or Ts.shape != Ys.shape:
        raise ValueError("T and Y must be ints or 1-D arrays of equal length")
    Q = len(Ts)
    cx = torch.as_tensor(np.asarray(ctx_x), dtype=torch.float32)[None]
    ci = torch.as_tensor(np.asarray(ctx_int), dtype=torch.bool)[None]
    D = cx.shape[-1]
    if normalise:  # same per-task standardisation as prepare_mixed_batch
        mean, std = _normalisation_stats(cx, ci)
        mean, std = mean[0], std[0]
    else:
        mean, std = torch.zeros(D), torch.ones(D)
    cxn = ((cx - mean) / std)[..., None].repeat(Q, 1, 1, 1)
    # (log std, mean / std) per variable, as in prepare_mixed_batch
    scale = torch.stack([std.log(), mean / std], dim=-1)[None].repeat(Q, 1, 1)
    flags = (torch.zeros_like(ci) if zero_flags else ci).repeat(Q, 1, 1)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32)
    tgt = torch.zeros(Q, len(tt), D, 1)
    for q, Tq in enumerate(Ts.tolist()):
        tgt[q, :, Tq, 0] = (tt - mean[Tq]) / std[Tq]
    inp = dict(
        context_data=cxn,
        target_data=tgt,
        intervention_index=Ts,
        outcome_index=Ys,
        context_int_mask=flags,
        context_scale=scale,
    )
    return inp, mean, std


def _call(model: torch.nn.Module, inp: dict, device):
    """Run the model on `_model_inputs` output, cast to its device/dtype."""
    dt = next(model.parameters()).dtype  # match the model (float32 or bfloat16)
    return model(
        context_data=inp["context_data"].to(device, dt),
        target_data=inp["target_data"].to(device, dt),
        intervention_index=inp["intervention_index"].to(device),
        outcome_index=inp["outcome_index"].to(device),
        context_int_mask=inp["context_int_mask"].to(device),
        context_scale=inp["context_scale"].to(device, dt),
    )


@torch.no_grad()
def predict_mixture(
    model: torch.nn.Module,
    ctx_x: np.ndarray,
    ctx_int: np.ndarray,
    T: int | np.ndarray,
    Y: int | np.ndarray,
    t: np.ndarray,
    *,
    device: str | torch.device,
    zero_flags: bool = False,
    normalise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Model predictive p(X_Y | do(X_T = t), context) on the RAW data scale.

    ctx_x [N, D] float, ctx_int [N, D] bool: one context.
    T, Y: int or 1-D int arrays of equal length Q (batch of queries sharing the context).
    t: 1-D array of do-values [n_t], shared by all queries.
    Returns mu, sd, logw as float64 tensors [Q, n_t, K] on `device`; logw normalised
    over K. With scalar T, Y, Q = 1.

    The context is standardised per task with `_normalisation_stats` (as in
    training) when `normalise`, else left as is; zero_flags sets every
    intervened-cell flag to False (flag ablation).
    """
    inp, mean, std = _model_inputs(
        ctx_x, ctx_int, T, Y, t, zero_flags=zero_flags, normalise=normalise
    )
    out = _call(model, inp, device)
    return _mixture_on_raw_scale(out, inp["outcome_index"], mean, std, device)


def _mixture_on_raw_scale(out, outcome_index, mean, std, device):
    """(mu, sd, logw) float64 [B, n_t, K] on the raw data scale from a model
    output on the standardised scale; `mean`, `std` [D] as returned by
    `_model_inputs`, `outcome_index` [B]. logw is normalised over K."""
    Ys = outcome_index
    mu = out.pred_mean[:, :, 0].double()  # [Q, n_t, K], standardised scale
    sd = out.pred_std[:, :, 0].double()
    w = out.weights[:, :, 0].double()
    m_y = mean[Ys].double().to(device)[:, None, None]
    s_y = std[Ys].double().to(device)[:, None, None]
    logw = torch.log(w + 1e-12)
    logw = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
    return mu * s_y + m_y, sd * s_y + 1e-12, logw


def _as_like(y, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(y), dtype=ref.dtype, device=ref.device)


def mixture_logpdf(
    mu: torch.Tensor, sd: torch.Tensor, logw: torch.Tensor, y
) -> torch.Tensor:
    """log density of y under the mixtures; mu, sd, logw [..., K], y
    broadcastable to [...] (e.g. [n_t] against [Q, n_t, K]) -> [...]."""
    y = _as_like(y, mu)[..., None]
    lp = torch.distributions.Normal(mu, sd).log_prob(y)
    return torch.logsumexp(lp + logw, dim=-1)


def mixture_cdf(
    mu: torch.Tensor, sd: torch.Tensor, logw: torch.Tensor, y
) -> torch.Tensor:
    """Mixture CDF F(y), i.e. PIT values for observed y; shapes as in
    mixture_logpdf."""
    y = _as_like(y, mu)[..., None]
    return (logw.exp() * torch.distributions.Normal(mu, sd).cdf(y)).sum(-1)


def effective_components(logw: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of the mixture weights [..., K] -> [...]."""
    w = logw.exp()
    return torch.exp(-(w * torch.log(w + 1e-12)).sum(-1))


# --------------------------------------------------------------------------- #
# Internal representations
# --------------------------------------------------------------------------- #
def hidden_state_layout(model: torch.nn.Module) -> dict:
    """How the instantiated model lays out its hidden tokens.

    Returns a dict with
    - architecture: "extra_node", "layerwise", "r1", "reg_axis" or
      "upstream", read from the encoder class (not from file names);
    - share_intervention_encoder: for "layerwise", whether the injection
      reuses `context_intervened` (None for other architectures);
    - metadata_token: True if every row carries one extra node position
      after the D variables (extra-node), so the hidden states have D + 1
      node positions;
    - sample_attn_mode: "MHSA" or "MHCA".
    """
    from CITNP.models.extra_node_encoder import ExtraNodeMixedCausalInfEncoder
    from CITNP.models.mixed_context_model import (
        LayerwiseMixedCausalInfEncoder,
        MixedCausalInfEncoder,
    )
    from CITNP.models.regime_axis.encoder import RegimeAxisEncoder

    enc = model.encoder
    share = None
    if isinstance(enc, ExtraNodeMixedCausalInfEncoder):
        arch = "extra_node"
    elif isinstance(enc, LayerwiseMixedCausalInfEncoder):
        arch = "layerwise"
        share = bool(enc.share_intervention_encoder)
    elif isinstance(enc, MixedCausalInfEncoder):
        arch = "r1"
    elif isinstance(enc, RegimeAxisEncoder):
        arch = "reg_axis"
    else:
        arch = "upstream"
    return dict(
        architecture=arch,
        share_intervention_encoder=share,
        metadata_token=arch == "extra_node",
        sample_attn_mode=getattr(enc.encoder, "sample_attn_mode", None),
    )


def probe_points(model: torch.nn.Module) -> list[str]:
    """Names of the depths `context_states` returns, in forward order:
    "embed" (encoder input), then "block<l>/sample" and "block<l>/node"
    (after the sample- and node-attention layer of encoder block l)."""
    stack = model.encoder.encoder
    names = ["embed"]
    for i in range(len(stack.node_layers)):
        names += [f"block{i}/sample", f"block{i}/node"]
    return names


def _record_states(
    model: torch.nn.Module,
    inp: dict,
    device,
    *,
    keep: Callable[[torch.Tensor], Any] = lambda x: x,
) -> tuple[dict[str, dict[str, Any]], Any]:
    """Run the model once on prepared inputs (`_model_inputs` output, any
    batch size B) and record the residual stream of every token at each
    depth of `probe_points(model)`.

    Hook locations: "embed" is the input of the transformer stack (role
    embeddings, flag or metadata embedding and scale conditioning included;
    for layer-wise models it is before `inject_sample[0]`). "block<l>/sample"
    and "block<l>/node" are the outputs of the pre-norm sample- and
    node-attention layers of block l (after both residual additions,
    attention and feed-forward), i.e. before any later re-injection. With
    MHCA, context rows are read from the self-attention layer and query rows
    from the cross-attention layer of the block.

    Returns ({stage: {"context": keep(x_ctx), "query": keep(x_qry)}}, model
    output), where x_ctx is [B, Nc, D', d_model] and x_qry [B, Nq, D',
    d_model] with every internal node position kept (D' = D + 1 for the
    extra-node metadata token). `keep` maps each recorded tensor (on the
    model's device) to what is stored. Hooks are removed even on errors.
    """
    stack = model.encoder.encoder
    B, Nc = inp["context_data"].shape[:2]
    states: dict[str, dict[str, Any]] = {n: {} for n in probe_points(model)}

    def store(name, part, x):  # x [B, rows, D', d_model]
        states[name][part] = keep(x.detach())

    def pre_embed(_mod, args):
        store("embed", "context", args[0][:, :Nc])
        store("embed", "query", args[0][:, Nc:])

    def after_sample(name, part):  # part: "context", "query" or "both"
        def hook(_mod, _args, out):  # out [B * D', S, d_model]
            S = out.shape[1]
            x = out.reshape(B, -1, S, out.shape[-1]).permute(0, 2, 1, 3)
            if part == "both":
                store(name, "context", x[:, :Nc])
                store(name, "query", x[:, Nc:])
            else:
                store(name, part, x)

        return hook

    def after_node(name):
        def hook(_mod, _args, out):  # out [B * S, D', d_model]
            x = out.reshape(B, -1, *out.shape[1:])
            store(name, "context", x[:, :Nc])
            store(name, "query", x[:, Nc:])

        return hook

    handles = [stack.register_forward_pre_hook(pre_embed)]
    try:
        mhca = stack.sample_attn_mode == "MHCA"
        for i, node_layer in enumerate(stack.node_layers):
            name = f"block{i}/sample"
            if mhca:
                # sample_layers = [mhsa_0, mhca_0, mhsa_1, ...]: context rows
                # pass through mhsa_i only, query rows through mhca_i only.
                handles.append(stack.sample_layers[2 * i].register_forward_hook(
                    after_sample(name, "context")))  # fmt: skip
                handles.append(stack.sample_layers[2 * i + 1].register_forward_hook(
                    after_sample(name, "query")))  # fmt: skip
            else:
                handles.append(stack.sample_layers[i].register_forward_hook(
                    after_sample(name, "both")))  # fmt: skip
            handles.append(
                node_layer.register_forward_hook(after_node(f"block{i}/node"))
            )
        out = _call(model, inp, device)
    finally:
        for h in handles:
            h.remove()
    return states, out


@torch.no_grad()
def context_states(
    model: torch.nn.Module,
    ctx_x: np.ndarray,
    ctx_int: np.ndarray,
    T: int,
    Y: int,
    *,
    device: str | torch.device,
    zero_flags: bool = False,
    normalise: bool = True,
    include_metadata_token: bool = False,
    include_query_states: bool = False,
    t: np.ndarray | None = None,
):
    """Residual-stream representation of every CONTEXT cell at each depth of
    `probe_points(model)`, for one context and one question (T, Y).

    Returns {name: float32 CPU tensor [N, D, d_model]}. Context rows attend
    only to context rows, so the states do not depend on the query do-values
    (a single dummy query point is used). The role embedding does depend on
    (T, Y), hence the arguments. zero_flags sets every intervened-cell flag to
    False, as in `predict_mixture`. Works for every mixed-context model class
    (the hooks sit on the upstream attention layers, which all share).
    Internal node positions beyond the D variables (e.g. the extra-node
    metadata column) are dropped. See `_record_states` for the exact hook
    locations.

    Options (defaults reproduce the outputs above exactly):
    - include_metadata_token: keep the extra-node metadata position, so the
      states are [N, D + 1, d_model] with the metadata token at index D
      (no effect for architectures without one, see `hidden_state_layout`).
    - include_query_states: also return the query rows; the result is then
      (context states, query states), the latter {name: [n_t, D or D + 1,
      d_model]}, one row per do-value in `t` (default: a single 0).
    """
    layout = hidden_state_layout(model)
    if (include_metadata_token or include_query_states) and layout[
        "architecture"
    ] not in ("extra_node", "layerwise", "r1", "upstream"):
        raise ValueError(
            f"include_metadata_token / include_query_states do not support "
            f"the {layout['architecture']!r} hidden-state layout"
        )
    D = int(np.asarray(ctx_x).shape[1])
    width = D + 1 if include_metadata_token and layout["metadata_token"] else D
    inp, _, _ = _model_inputs(
        ctx_x, ctx_int, int(T), int(Y), np.zeros(1) if t is None else t,
        zero_flags=zero_flags, normalise=normalise,
    )  # fmt: skip
    states, _ = _record_states(
        model, inp, device, keep=lambda x: x.float().cpu()[0, :, :width]
    )
    names = probe_points(model)
    ctx = {name: states[name]["context"] for name in names}
    if include_query_states:
        return ctx, {name: states[name]["query"] for name in names}
    return ctx
