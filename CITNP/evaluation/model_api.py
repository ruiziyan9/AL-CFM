"""The only code that calls a trained model: loading it, getting its
predictive mixture for interventional queries on the raw data scale, and
reading its internal context representations (`context_states`).

Everything else derives what it needs from `predict_mixture` via
`mixture_logpdf`, `mixture_cdf` and `effective_components`, or from
`context_states`.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.trainer.mixed_context_trainer import _normalisation_stats
from CITNP.utils.configs import CausalInfModelConfig

DEFAULT_MODEL_CLASS = "MixedContextMoGModel"
# Copy of the training prior config that mixed_train writes next to the
# checkpoints, with the original file's suffix (.yaml, .yml or .json).
TRAINING_PRIOR_CONFIG_STEM = "training_prior_config"
# Number of variables of a checkpoint that records none (the old default).
FALLBACK_NUM_NODES = 3
_INDEX_EMBEDDING_KEY = "encoder.index_embedding.weight"


def training_prior_config_path(ckpt: str | Path) -> Path | None:
    """The prior config the checkpoint `ckpt` was trained on, if its run
    directory (the checkpoint's directory) records it:
      1. <run dir>/training_prior_config.{yaml,yml,json}, the copy that
         mixed_train writes;
      2. <run dir>/presimulated_data.yaml (runs trained from pre-simulated
         data): the prior_config of its dataset_config, if both exist.
    None if neither is there (e.g. runs from before the copy was written).
    """
    run_dir = Path(ckpt).resolve().parent
    for suffix in (".yaml", ".yml", ".json"):
        path = run_dir / f"{TRAINING_PRIOR_CONFIG_STEM}{suffix}"
        if path.is_file():
            return path
    record = run_dir / "presimulated_data.yaml"
    if record.is_file():
        from CITNP.datasets.dataset_config import (
            load_presimulated_dataset_config,
            prior_config_path,
        )

        dataset_config = yaml.safe_load(record.read_text()).get("dataset_config")
        if dataset_config and Path(dataset_config).is_file():
            path = prior_config_path(
                load_presimulated_dataset_config(dataset_config)
            )
            if path.is_file():
                return path
    return None


def checkpoint_num_variables(
    ckpt: str | Path, state_dict: dict | None = None
) -> tuple[int | None, list[int] | None, str]:
    """What the checkpoint says about its number of variables.

    Returns (num_nodes, training_num_variables, source):
      num_nodes: D the model was built for (training builds it for the
          largest D of its prior): max(num_variables) of the training prior
          config (`training_prior_config_path`), else the number of rows of
          the extra-node index embedding in `state_dict`, else None;
      training_num_variables: the prior's num_variables, if the prior is
          known, else None;
      source: where num_nodes came from, for error messages.
    Raises ValueError if the prior config and the index embedding disagree.
    """
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    rows = None
    if state_dict is not None and _INDEX_EMBEDDING_KEY in state_dict:
        rows = int(state_dict[_INDEX_EMBEDDING_KEY].shape[0])
    prior_path = training_prior_config_path(ckpt)
    if prior_path is not None:
        prior = load_config(prior_path)
        num_variables = sorted(int(d) for d in prior.num_variables)
        if rows is not None and rows != max(num_variables):
            raise ValueError(
                f"{ckpt}: the extra-node index embedding has {rows} rows but "
                f"the training prior {prior_path} has num_variables "
                f"{num_variables}"
            )
        return max(num_variables), num_variables, f"training prior {prior_path}"
    if rows is not None:
        return rows, None, f"{ckpt} (extra-node index embedding)"
    return None, None, "nothing recorded"


def load_model(
    ckpt: str, model_config: str, device: str, num_nodes: int | None = None
) -> torch.nn.Module:
    """Build the model described by the YAML `model_config`, load the
    weights in `ckpt` and return it in eval mode, ready for data with
    `num_nodes` variables.

    num_nodes (the evaluation's D) defaults to the number of variables the
    checkpoint records (`checkpoint_num_variables`: largest D of the
    training prior config, or the extra-node index embedding), else
    FALLBACK_NUM_NODES = 3, as before.

    Only an architecture that cannot run at num_nodes is an error: an
    extra-node model has one index embedding per variable, so num_nodes
    larger than its index-embedding table raises ValueError (it is built for
    the table's size and accepts fewer variables). The other encodings have
    no parameter that depends on D: they are built for num_nodes (or the
    recorded D, if larger) and run at any D. If num_nodes is outside the
    training prior's num_variables, a UserWarning says so and loading
    continues.

    The class is resolved by resolve_model_class from the optional YAML keys
    intervention_encoding and/or model_class (default: r1,
    MixedContextMoGModel).
    """
    from CITNP.models.mixed_context_model import resolve_model_class
    from CITNP.models.regime_axis import REG_AXIS_CONFIG_KEYS

    state = torch.load(ckpt, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    built, training_num_variables, source = checkpoint_num_variables(
        ckpt, state_dict
    )
    if num_nodes is None:
        num_nodes = built if built is not None else FALLBACK_NUM_NODES
    index_rows = None
    if _INDEX_EMBEDDING_KEY in state_dict:
        index_rows = int(state_dict[_INDEX_EMBEDDING_KEY].shape[0])
    if index_rows is not None and num_nodes > index_rows:
        raise ValueError(
            f"num_variables={num_nodes} cannot run on this checkpoint: its "
            f"extra-node index embedding has {index_rows} rows, one per "
            f"variable ({ckpt})"
        )
    if training_num_variables is not None and (
        num_nodes not in training_num_variables
    ):
        warnings.warn(
            f"num_variables={num_nodes} is outside the training prior's "
            f"num_variables {training_num_variables} ({source}); the model "
            "runs, but was not trained on this D",
            stacklevel=2,
        )
    # extra_node: the table's size; otherwise any size works
    build_size = index_rows if index_rows is not None else max(
        num_nodes, built or 0
    )
    with open(model_config) as f:
        m = yaml.safe_load(f)
    cfg = CausalInfModelConfig(
        d_model=m["d_model"],
        emb_depth=m["emb_depth"],
        dim_feedforward=m["dim_feedforward"],
        nhead=m["nhead"],
        dropout=0.0,
        num_layers_encoder=m["num_layers_encoder"],
        num_nodes=build_size,
        sample_attn_mode=m["sample_attn_mode"],
        linear_attention=m.get("linear_attention", False),
        mean_loss_across_samples=False,
        device=device,
        dtype="float32",
        num_mixture_components=m["num_mixture_components"],
        scale_conditioning=m.get("scale_conditioning", "none"),
        # reg_axis only; absent keys keep the dataclass defaults
        **{k: m[k] for k in REG_AXIS_CONFIG_KEYS if k in m},
    )
    model_cls, _ = resolve_model_class(
        m.get("intervention_encoding"), m.get("model_class")
    )
    model = model_cls(cfg)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _model_inputs(
    ctx_x, ctx_int, T, Y, t, *, zero_flags, normalise, normalisation_int_mask=None
):
    """Model keyword inputs (CPU, float32) for one context and Q queries,
    plus the per-variable standardisation (mean, std) [D] that was applied.
    `context_scale` [Q, D, 2] is (log std, mean / std) per variable (zeros
    without normalisation), used by models with scale_conditioning "stats".
    The standardisation uses the observational rows of
    `normalisation_int_mask` [N, D] (default: `ctx_int`). t is [n_t] (shared
    by all queries) or [Q, n_t] (one row of do-values per query)."""
    Ts = torch.as_tensor(np.atleast_1d(T), dtype=torch.long)
    Ys = torch.as_tensor(np.atleast_1d(Y), dtype=torch.long)
    if Ts.ndim != 1 or Ts.shape != Ys.shape:
        raise ValueError("T and Y must be ints or 1-D arrays of equal length")
    Q = len(Ts)
    cx = torch.as_tensor(np.asarray(ctx_x), dtype=torch.float32)[None]
    ci = torch.as_tensor(np.asarray(ctx_int), dtype=torch.bool)[None]
    D = cx.shape[-1]
    if normalise:  # same per-task standardisation as prepare_mixed_batch
        norm_mask = ci
        if normalisation_int_mask is not None:
            norm_mask = torch.as_tensor(
                np.asarray(normalisation_int_mask), dtype=torch.bool
            )[None]
        mean, std = _normalisation_stats(cx, norm_mask)
        mean, std = mean[0], std[0]
    else:
        mean, std = torch.zeros(D), torch.ones(D)
    cxn = ((cx - mean) / std)[..., None].repeat(Q, 1, 1, 1)
    # (log std, mean / std) per variable, as in prepare_mixed_batch
    scale = torch.stack([std.log(), mean / std], dim=-1)[None].repeat(Q, 1, 1)
    flags = (torch.zeros_like(ci) if zero_flags else ci).repeat(Q, 1, 1)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32)
    if tt.ndim == 1:  # do-values shared by every query
        tt = tt[None].expand(Q, -1)
    if tt.ndim != 2 or tt.shape[0] != Q:
        raise ValueError("t must be [n_t] or [Q, n_t] (one row per query)")
    tgt = torch.zeros(Q, tt.shape[1], D, 1)
    for q, Tq in enumerate(Ts.tolist()):
        tgt[q, :, Tq, 0] = (tt[q] - mean[Tq]) / std[Tq]
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
    normalisation_int_mask: np.ndarray | None = None,
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

    normalisation_int_mask [N, D] bool, optional: take the standardisation
    from the observational rows of this mask instead of `ctx_int`, e.g. to
    edit the intervention mask without changing how the values are scaled.
    Everything derived from the standardisation follows it: the standardised
    context and do-values, the scale features `context_scale` (log std,
    mean / std) of scale_conditioning "stats" models, and the map of the
    prediction back to the raw scale. Default None: the standardisation of
    `ctx_int`, as in training.
    """
    inp, mean, std = _model_inputs(
        ctx_x, ctx_int, T, Y, t, zero_flags=zero_flags, normalise=normalise,
        normalisation_int_mask=normalisation_int_mask,
    )  # fmt: skip
    Ys = inp["outcome_index"]
    out = _call(model, inp, device)
    mu = out.pred_mean[:, :, 0].double()  # [Q, n_t, K], standardised scale
    sd = out.pred_std[:, :, 0].double()
    w = out.weights[:, :, 0].double()
    m_y = mean[Ys].double().to(device)[:, None, None]
    s_y = std[Ys].double().to(device)[:, None, None]
    logw = torch.log(w + 1e-12)
    logw = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
    return mu * s_y + m_y, sd * s_y + 1e-12, logw


@torch.no_grad()
def predict_mixture_versions(
    model: torch.nn.Module,
    ctx_x: np.ndarray,
    ctx_ints: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    t: np.ndarray,
    *,
    device: str | torch.device,
    normalise: bool = True,
    normalisation_int_masks: np.ndarray | None = None,
    max_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`predict_mixture` for several intervention masks ("versions") of ONE
    context, in a single model forward pass.

    ctx_x [N, D]: the context values, shared by every version.
    ctx_ints [V, N, D] bool: one intervention mask per version.
    T, Y [Q] int: the queries, asked of every version.
    t [Q, n_t] (or [n_t], shared): do-values, one row per query.
    normalisation_int_masks [V, N, D] bool, optional: the mask whose
    observational rows standardise each version (default: its own mask, as
    `predict_mixture`; see there for everything that follows it).
    max_batch_size: split the V * Q batch elements into forward passes of at
    most this many (default: one pass), to bound memory.

    Returns mu, sd, logw as float64 tensors [V, Q, n_t, K] on `device`, on
    the raw scale, logw normalised over K; equal to `predict_mixture` per
    version up to the rounding of batched kernels.
    """
    ctx_ints = np.asarray(ctx_ints, dtype=bool)
    V = ctx_ints.shape[0]
    inputs, means, stds = [], [], []
    for v in range(V):
        inp, mean, std = _model_inputs(
            ctx_x, ctx_ints[v], T, Y, t, zero_flags=False, normalise=normalise,
            normalisation_int_mask=None
            if normalisation_int_masks is None
            else normalisation_int_masks[v],
        )  # fmt: skip
        inputs.append(inp)
        means.append(mean)
        stds.append(std)
    batch = {k: torch.cat([inp[k] for inp in inputs]) for k in inputs[0]}
    total = batch["intervention_index"].shape[0]
    step = total if max_batch_size is None else int(max_batch_size)
    parts: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, total, step):
        out = _call(model, {k: v[start : start + step] for k, v in batch.items()},
                    device)  # fmt: skip
        parts.append((out.pred_mean[:, :, 0], out.pred_std[:, :, 0],
                      out.weights[:, :, 0]))  # fmt: skip
    mu, sd, w = (torch.cat(p).double() for p in zip(*parts))  # [V * Q, n_t, K]
    Q = len(inputs[0]["outcome_index"])
    mu, sd, w = (p.reshape(V, Q, *p.shape[1:]) for p in (mu, sd, w))
    Ys = inputs[0]["outcome_index"]
    m_y = torch.stack([m[Ys] for m in means]).double().to(mu.device)
    s_y = torch.stack([s[Ys] for s in stds]).double().to(mu.device)
    m_y, s_y = m_y[:, :, None, None], s_y[:, :, None, None]
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
def uses_metadata_node(model: torch.nn.Module) -> bool:
    """True if `model` uses the extra-node intervention encoding, whose
    transformer runs on D + 1 node positions per row: the D variables and
    one intervention-metadata column (see CITNP/models/extra_node_encoder.py).
    """
    from CITNP.models.extra_node_encoder import ExtraNodeMixedCausalInfEncoder

    return isinstance(
        getattr(model, "encoder", None), ExtraNodeMixedCausalInfEncoder
    )


def _uses_regime_axis(model: torch.nn.Module) -> bool:
    """True if `model` uses the reg_axis encoding (RegimeAxisEncoder), which
    has its own blocks instead of the upstream transformer stack."""
    from CITNP.models.regime_axis import RegimeAxisEncoder

    return isinstance(getattr(model, "encoder", None), RegimeAxisEncoder)


def probe_points(model: torch.nn.Module) -> list[str]:
    """Names of the depths `context_states` returns, in forward order:
    "embed" (encoder input), then "block<l>/sample" and "block<l>/node"
    (after the sample- and node-attention layer of encoder block l).

    reg_axis models have no node-attention layer after their sample
    attention; for them the names are "embed", then per block l
    "block<l>/sample" (after the within-regime sample attention) and, in
    every block but the last, "block<l>/feedback" (after the context cells
    read the regime memories)."""
    if _uses_regime_axis(model):
        names = ["embed"]
        for i, block in enumerate(model.encoder.blocks):
            names.append(f"block{i}/sample")
            if block.feedback is not None:
                names.append(f"block{i}/feedback")
        return names
    stack = model.encoder.encoder
    names = ["embed"]
    for i in range(len(stack.node_layers)):
        names += [f"block{i}/sample", f"block{i}/node"]
    return names


def _register_upstream_stack_hooks(model, keep, num_context: int) -> list:
    """Hooks on the upstream transformer stack (r1, layerwise, extra_node)
    that pass every probe point's context states [1, Nc, D', d_model] to
    keep(name, states); returns the hook handles."""
    stack = model.encoder.encoder
    batch_size = 1

    def pre_embed(_mod, args):
        keep("embed", args[0][:, :num_context])

    def after_sample(name, ctx_only):
        def hook(_mod, _args, out):  # out [B * D', S, d_model]
            S = out.shape[1]
            x = out.reshape(batch_size, -1, S, out.shape[-1]).permute(0, 2, 1, 3)
            keep(name, x if ctx_only else x[:, :num_context])

        return hook

    def after_node(name):
        def hook(_mod, _args, out):  # out [B * S, D', d_model]
            x = out.reshape(batch_size, -1, *out.shape[1:])
            keep(name, x[:, :num_context])

        return hook

    handles = [stack.register_forward_pre_hook(pre_embed)]
    mhca = stack.sample_attn_mode == "MHCA"
    for i, node_layer in enumerate(stack.node_layers):
        # MHCA: sample_layers = [mhsa_0, mhca_0, mhsa_1, ...]; the context
        # rows only pass through the self-attention layer mhsa_i.
        sample_layer = stack.sample_layers[2 * i if mhca else i]
        handles.append(
            sample_layer.register_forward_hook(
                after_sample(f"block{i}/sample", ctx_only=mhca)
            )
        )
        handles.append(
            node_layer.register_forward_hook(after_node(f"block{i}/node"))
        )
    return handles


def _register_regime_axis_hooks(model, keep) -> list:
    """Hooks on the reg_axis blocks that pass every probe point's context
    states [1, Nc, D, d_model] to keep(name, states); returns the handles."""
    blocks = model.encoder.blocks

    def pre_embed(_mod, args):  # args[0]: context tokens [B, Nc, D, d_model]
        keep("embed", args[0])

    def after_sample(name):
        def hook(_mod, _args, out):  # out [B, D, Nc, d_model]
            keep(name, out.permute(0, 2, 1, 3))

        return hook

    def after_feedback(name):
        def hook(_mod, _args, out):  # out [B * D, Nc, d_model]
            x = out.reshape(1, -1, *out.shape[1:]).permute(0, 2, 1, 3)
            keep(name, x)

        return hook

    handles = [blocks[0].register_forward_pre_hook(pre_embed)]
    for i, block in enumerate(blocks):
        handles.append(
            block.sample_attn.register_forward_hook(
                after_sample(f"block{i}/sample")
            )
        )
        if block.feedback is not None:
            handles.append(
                block.feedback.register_forward_hook(
                    after_feedback(f"block{i}/feedback")
                )
            )
    return handles


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
    include_metadata_node: bool = False,
) -> dict[str, torch.Tensor]:
    """Residual-stream representation of every CONTEXT cell at each depth of
    `probe_points(model)`, for one context and one question (T, Y).

    Returns {name: float32 CPU tensor [N, D, d_model]}. Context rows attend
    only to context rows, so the states do not depend on the query do-values
    (a single dummy query point is used). The role embedding does depend on
    (T, Y), hence the arguments. zero_flags sets every intervened-cell flag to
    False, as in `predict_mixture`. Works for every mixed-context model class
    (r1, layerwise and extra_node: hooks on the upstream attention layers,
    which they all share; reg_axis: hooks on its blocks, see `probe_points`).

    Internal node positions beyond the D variables (the extra-node metadata
    column) are dropped, unless include_metadata_node is True: then the
    states are [N, D + 1, d_model] with the metadata column last. Only
    extra-node models have that column; for any other model
    include_metadata_node=True raises ValueError.
    """
    if include_metadata_node and not uses_metadata_node(model):
        raise ValueError(
            "include_metadata_node=True needs a model with the extra_node "
            f"intervention encoding; {type(model).__name__} has no metadata "
            "column"
        )
    Nc = int(np.asarray(ctx_x).shape[0])
    D = int(np.asarray(ctx_x).shape[1])
    kept_columns = D + 1 if include_metadata_node else D
    states: dict[str, torch.Tensor] = {}

    def keep(name, x):  # x [1, Nc, D', d_model]; D' = D + internal positions
        states[name] = x.detach().float().cpu()[0, :, :kept_columns]

    if _uses_regime_axis(model):
        handles = _register_regime_axis_hooks(model, keep)
    else:
        handles = _register_upstream_stack_hooks(model, keep, Nc)
    try:
        inp, _, _ = _model_inputs(
            ctx_x, ctx_int, int(T), int(Y), np.zeros(1),
            zero_flags=zero_flags, normalise=normalise,
        )  # fmt: skip
        _call(model, inp, device)
    finally:
        for h in handles:
            h.remove()
    return {name: states[name] for name in probe_points(model)}
