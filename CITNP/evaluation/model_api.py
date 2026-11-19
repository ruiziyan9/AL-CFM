"""The only code that calls a trained model: loading it, getting its
predictive mixture for interventional queries on the raw data scale, and
reading its internal context representations (`context_states`).

Everything else derives what it needs from `predict_mixture` via
`mixture_logpdf`, `mixture_cdf` and `effective_components`, or from
`context_states`.

Joint models (prediction_head: joint) predict all non-intervened variables
at once: `predict_joint_mixture` returns the joint mixture on the raw scale
(`JointPrediction`), and `joint_marginal_mixture`,
`joint_mixture_mean_and_covariance`, `sample_joint_rows` and
`joint_log_density` derive from it. `predict_mixture` works for joint models
too (through `joint_marginal_mixture`), so every marginal evaluation runs on
a joint checkpoint unchanged.
"""

from dataclasses import dataclass

import numpy as np
import torch

from CITNP.trainer.mixed_context_trainer import _normalisation_stats
from CITNP.utils.config_files import load_config_mapping
from CITNP.utils.configs import CausalInfModelConfig
from CITNP.utils.loss import joint_mixture_log_density
from CITNP.utils.outputs import JointModelOutput

DEFAULT_MODEL_CLASS = "LayerwiseMixedContextMoGModel"


def load_model(
    ckpt: str, model_config: str, device: str, num_nodes: int = 3
) -> torch.nn.Module:
    """Build the model described by the YAML `model_config` (with everything
    it inherits through `extends`, CITNP/utils/config_files.py) for
    `num_nodes` variables (`build_model`), load the weights in `ckpt` and
    return it in eval mode.

    A layer-wise run saved before share_intervention_encoder existed has a
    separate flag network but no entry in its resolved_model_config.yaml;
    loading it raises a ValueError that says to add
    `share_intervention_encoder: false` to that file.
    """
    model = build_model(model_config, device, num_nodes)
    state = torch.load(ckpt, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    _check_separate_flag_network(
        model, state_dict, load_config_mapping(model_config), model_config
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_model(
    model_config: str, device: str, num_nodes: int = 3
) -> torch.nn.Module:
    """The randomly initialised model described by the YAML `model_config`
    for `num_nodes` variables (the construction step of `load_model`).

    The class is resolved by resolve_model_class from the optional YAML keys
    intervention_encoding, model_class and prediction_head (default:
    layerwise, LayerwiseMixedContextMoGModel, marginal), exactly as in
    mixed_train.py. share_intervention_encoder absent means True (the shared
    flag network). The joint_* keys are checked as in mixed_train.py (given
    with a marginal head: ValueError).
    """
    from CITNP.models.mixed_context_model import resolve_model_class
    from CITNP.models.prediction_heads import check_joint_head_settings
    from CITNP.models.regime_axis import REG_AXIS_CONFIG_KEYS

    m = load_config_mapping(model_config)
    model_cls, _ = resolve_model_class(
        m.get("intervention_encoding"),
        m.get("model_class"),
        m.get("prediction_head"),
    )
    is_joint = getattr(model_cls, "prediction_head", "marginal") == "joint"
    joint_settings = check_joint_head_settings(
        m, "joint" if is_joint else "marginal", m["num_mixture_components"]
    )
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
        # reg_axis / reg_axis_full only; absent keys keep the dataclass
        # defaults
        **{k: m[k] for k in REG_AXIS_CONFIG_KEYS if k in m},
        # layerwise only; absent = the shared flag network (default)
        share_intervention_encoder=bool(
            m.get("share_intervention_encoder", True)
        ),
        prediction_head="joint" if is_joint else "marginal",
        **joint_settings,
    )
    return model_cls(cfg)


SEPARATE_FLAG_NETWORK_PREFIX = "encoder.flag_code."


def _check_separate_flag_network(
    model: torch.nn.Module, state_dict: dict, settings: dict, model_config: str
) -> None:
    """Raise a ValueError for a layer-wise checkpoint with a separate flag
    network (`encoder.flag_code.*`) whose model config does not name
    share_intervention_encoder: such a run was saved before the key existed,
    when absent meant separate, and the key now defaults to shared."""
    if "share_intervention_encoder" in settings:
        return
    has_separate = any(
        k.startswith(SEPARATE_FLAG_NETWORK_PREFIX) for k in state_dict
    )
    model_has_separate = any(
        k.startswith(SEPARATE_FLAG_NETWORK_PREFIX) for k in model.state_dict()
    )
    if has_separate and not model_has_separate:
        raise ValueError(
            "The checkpoint has a separate flag network (encoder.flag_code.*), "
            f"but {model_config} does not set share_intervention_encoder, "
            "which now defaults to true (shared). This run was saved before "
            "the key existed: add `share_intervention_encoder: false` to its "
            "resolved_model_config.yaml (or to the model config passed here)."
        )


def _model_inputs(ctx_x, ctx_int, T, Y, t, *, zero_flags, normalise):
    """Model keyword inputs (CPU, float32) for one context and Q queries,
    plus the per-variable standardisation (mean, std) [D] that was applied.
    `context_scale` [Q, D, 2] is (log std, mean / std) per variable (zeros
    without normalisation), used by models with scale_conditioning "stats".

    Y=None (joint models): Q is the length of T and no outcome index is
    built (`outcome_index` is None)."""
    Ts = torch.as_tensor(np.atleast_1d(T), dtype=torch.long)
    if Y is None:
        if Ts.ndim != 1:
            raise ValueError("T must be an int or a 1-D array")
        Ys = None
    else:
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
    outcome_index = inp["outcome_index"]
    return model(
        context_data=inp["context_data"].to(device, dt),
        target_data=inp["target_data"].to(device, dt),
        intervention_index=inp["intervention_index"].to(device),
        outcome_index=None if outcome_index is None else outcome_index.to(device),
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

    For a joint model the marginals are read from the joint prediction: each
    distinct treatment in T is predicted once (`predict_joint_mixture`) and
    `joint_marginal_mixture` takes the requested Y, so queries with the same
    T and different Y read the same joint prediction. An intervened Y
    (Y == T) raises.
    """
    if getattr(model, "prediction_head", "marginal") == "joint":
        return _marginal_from_joint_model(
            model, ctx_x, ctx_int, T, Y, t,
            device=device, zero_flags=zero_flags, normalise=normalise,
        )  # fmt: skip
    inp, mean, std = _model_inputs(
        ctx_x, ctx_int, T, Y, t, zero_flags=zero_flags, normalise=normalise
    )
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


# --------------------------------------------------------------------------- #
# Joint models: p(X_{-T} | do(X_T = t), context)
# --------------------------------------------------------------------------- #
@dataclass
class JointPrediction:
    """The joint predictive mixture of a joint model for Q queries (one
    treatment each) at n_t do-values, on the RAW data scale, float64:

        p(x) = sum_k exp(log_weights_k) N(x; component_mean_k, Sigma_k),
        Sigma_k = covariance_factor_k covariance_factor_k^T
                  + diag(covariance_diagonal_k).

    All D variables are kept; the entries of the intervened variable are
    auxiliary (not a prediction: the variable is fixed at its do-value), and
    the distribution of interest is the block of the non-intervened ones.

    Attributes:
    -----------
    - component_mean: [Q, n_t, K, D].
    - covariance_factor: [Q, n_t, K, D, R].
    - covariance_diagonal: [Q, n_t, K, D].
    - log_weights: [Q, n_t, K], normalised over K.
    - intervened_mask: [Q, D] bool, True = set by do().
    """

    component_mean: torch.Tensor
    covariance_factor: torch.Tensor
    covariance_diagonal: torch.Tensor
    log_weights: torch.Tensor
    intervened_mask: torch.Tensor

    def select(self, queries: torch.Tensor) -> "JointPrediction":
        """The predictions of the given query positions [Q'] (long)."""
        return JointPrediction(
            component_mean=self.component_mean[queries],
            covariance_factor=self.covariance_factor[queries],
            covariance_diagonal=self.covariance_diagonal[queries],
            log_weights=self.log_weights[queries],
            intervened_mask=self.intervened_mask[queries],
        )


@torch.no_grad()
def predict_joint_mixture(
    model: torch.nn.Module,
    ctx_x: np.ndarray,
    ctx_int: np.ndarray,
    T: int | np.ndarray,
    t: np.ndarray,
    *,
    device: str | torch.device,
    zero_flags: bool = False,
    normalise: bool = True,
) -> JointPrediction:
    """Joint predictive p(X_{-T} | do(X_T = t), context) of a joint model on
    the RAW data scale.

    ctx_x [N, D] float, ctx_int [N, D] bool: one context. T: int or 1-D int
    array [Q] (one treatment per query). t: do-values [n_t], shared by all
    queries. Standardisation and zero_flags as in `predict_mixture`.

    The model's standardised parameters are un-standardised with the
    per-variable (mean, std), along the variable axis D: means * std + mean,
    factor rows * std, diagonal * std**2. The parameters are then on the raw
    scale, so densities computed from them need no separate Jacobian term.

    Returns a `JointPrediction` (float64 tensors on `device`).
    """
    inp, mean, std = _model_inputs(
        ctx_x, ctx_int, T, None, t, zero_flags=zero_flags, normalise=normalise
    )
    out = _call(model, inp, device)
    if not isinstance(out, JointModelOutput):
        raise ValueError(
            "predict_joint_mixture needs a joint model (prediction_head: joint)"
        )
    m = mean.double().to(device)  # [D]
    s = std.double().to(device)  # [D]
    return JointPrediction(
        component_mean=out.component_mean.double() * s + m,
        covariance_factor=out.covariance_factor.double() * s[:, None],
        covariance_diagonal=out.covariance_diagonal.double() * s**2,
        log_weights=torch.log_softmax(out.mixture_logits.double(), dim=-1),
        intervened_mask=out.intervened_mask.to(device),
    )


def joint_marginal_mixture(
    joint_prediction: JointPrediction, Y: int | np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The univariate marginal of variable Y of each query's joint mixture,
    as (mu, sd, logw) [Q, n_t, K], identical in meaning to `predict_mixture`:

        mu = component_mean[..., Y],
        sd = sqrt(sum_r covariance_factor[..., Y, r]**2
                  + covariance_diagonal[..., Y]),
        logw = log_weights.

    Y: int (the same variable for every query) or 1-D int array [Q]. Raises
    ValueError if Y is an intervened variable of its query: a variable fixed
    by do() has no predictive distribution here.
    """
    Q = joint_prediction.log_weights.shape[0]
    device = joint_prediction.log_weights.device
    Ys = torch.as_tensor(np.atleast_1d(Y), dtype=torch.long, device=device)
    if Ys.ndim != 1 or len(Ys) not in (1, Q):
        raise ValueError(f"Y must be an int or a 1-D array of length {Q}")
    Ys = Ys.expand(Q)
    q = torch.arange(Q, device=device)
    if joint_prediction.intervened_mask[q, Ys].any():
        raise ValueError(
            "Y is an intervened variable of its query; a variable fixed by "
            "do() has no predictive distribution"
        )
    # advanced indices on dims 0 and 3 -> [Q, n_t, K] (and [Q, n_t, K, R])
    mu = joint_prediction.component_mean[q, :, :, Ys]
    factor = joint_prediction.covariance_factor[q, :, :, Ys, :]
    diagonal = joint_prediction.covariance_diagonal[q, :, :, Ys]
    sd = torch.sqrt((factor**2).sum(-1) + diagonal)
    return mu, sd, joint_prediction.log_weights


def _marginal_from_joint_model(
    model, ctx_x, ctx_int, T, Y, t, *, device, zero_flags, normalise
):
    """`predict_mixture` for a joint model: each distinct treatment of T is
    predicted once (in one batch), and each query reads the marginal of its
    Y from its treatment's joint prediction."""
    Ts = np.atleast_1d(np.asarray(T)).astype(int)
    Ys = np.atleast_1d(np.asarray(Y)).astype(int)
    if Ts.ndim != 1 or Ts.shape != Ys.shape:
        raise ValueError("T and Y must be ints or 1-D arrays of equal length")
    treatments, position = np.unique(Ts, return_inverse=True)
    joint = predict_joint_mixture(
        model, ctx_x, ctx_int, treatments, t,
        device=device, zero_flags=zero_flags, normalise=normalise,
    )  # fmt: skip
    selected = joint.select(torch.as_tensor(position, device=device))
    return joint_marginal_mixture(selected, Ys)


def joint_mixture_mean_and_covariance(
    joint_prediction: JointPrediction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean [Q, n_t, D] and covariance [Q, n_t, D, D] of each whole mixture:

        mean = sum_k w_k mu_k,
        covariance = sum_k w_k (Sigma_k + mu_k mu_k^T) - mean mean^T.

    The rows and columns of the covariance and the mean entries of the
    intervened variables are set to zero (not predictions: those variables
    are fixed by do())."""
    jp = joint_prediction
    w = jp.log_weights.exp()  # [Q, n_t, K]
    mu = jp.component_mean  # [Q, n_t, K, D]
    factor = jp.covariance_factor
    outer_factor = factor @ factor.transpose(-1, -2)
    sigma = outer_factor + torch.diag_embed(jp.covariance_diagonal)
    outer_means = mu[..., :, None] * mu[..., None, :]  # [Q, n_t, K, D, D]
    mean = (w[..., None] * mu).sum(-2)  # [Q, n_t, D]
    second = (w[..., None, None] * (sigma + outer_means)).sum(-3)
    covariance = second - mean[..., :, None] * mean[..., None, :]
    keep = (~jp.intervened_mask).to(mean.dtype)[:, None]  # [Q, 1, D]
    mean = mean * keep
    covariance = covariance * keep[..., :, None] * keep[..., None, :]
    return mean, covariance


def sample_joint_rows(
    joint_prediction: JointPrediction,
    t: np.ndarray,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full rows [Q, n_t, num_samples, D] drawn from each query's joint
    mixture at each do-value, on the raw scale.

    One mixture component is drawn per row and used for all variables of
    that row (never a component per variable):
        k ~ Categorical(w), z ~ N(0, I_R), e ~ N(0, I_D),
        x = mu_k + F_k z + sqrt(v_k) * e.
    Then the intervened variable is overwritten with its do-value t; the
    marginal of a Gaussian over a subset of its coordinates is the
    corresponding block, so the non-intervened variables have the right
    distribution. `generator` must live on the prediction's device.
    """
    jp = joint_prediction
    Q, n_t, K, D = jp.component_mean.shape
    R = jp.covariance_factor.shape[-1]
    dtype, device = jp.component_mean.dtype, jp.component_mean.device
    weights = jp.log_weights.exp().reshape(-1, K)
    component = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    )
    component = component.reshape(Q, n_t, num_samples)
    index = component[..., None].expand(Q, n_t, num_samples, D)
    mu = jp.component_mean.gather(2, index)  # [Q, n_t, S, D]
    diagonal = jp.covariance_diagonal.gather(2, index)
    factor = jp.covariance_factor.gather(
        2, index[..., None].expand(Q, n_t, num_samples, D, R)
    )  # [Q, n_t, S, D, R]
    z = torch.randn(
        Q, n_t, num_samples, R, generator=generator, dtype=dtype, device=device
    )
    e = torch.randn(
        Q, n_t, num_samples, D, generator=generator, dtype=dtype, device=device
    )
    rows = mu + (factor @ z[..., None]).squeeze(-1) + diagonal.sqrt() * e
    do_values = torch.as_tensor(np.asarray(t), dtype=dtype, device=device)
    intervened = jp.intervened_mask[:, None, None, :]  # [Q, 1, 1, D]
    return torch.where(intervened, do_values[None, :, None, None], rows)


def joint_log_density(
    joint_prediction: JointPrediction, rows: torch.Tensor
) -> torch.Tensor:
    """Log density of the NON-INTERVENED variables of full rows under each
    query's joint mixture (raw scale, float64).

    It is a density over D minus the number of intervened variables, not
    over D variables: the intervened coordinates of `rows` are ignored
    (padding, as in `JointMixtureGaussianLoss`).

    rows: [Q, n_t, D], or [Q, n_t, S, D] with a sample axis.
    Returns [Q, n_t], or [Q, n_t, S].
    """
    jp = joint_prediction
    output = JointModelOutput(
        component_mean=jp.component_mean,
        covariance_factor=jp.covariance_factor,
        covariance_diagonal=jp.covariance_diagonal,
        mixture_logits=jp.log_weights,
        intervened_mask=jp.intervened_mask,
    )
    rows = torch.as_tensor(rows, device=jp.component_mean.device)
    if rows.ndim == 3:
        return joint_mixture_log_density(output, rows)
    # sample axis first, so the parameters [Q, n_t, ...] broadcast against it
    log_density = joint_mixture_log_density(output, rows.permute(2, 0, 1, 3))
    return log_density.permute(1, 2, 0)


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
def probe_points(model: torch.nn.Module) -> list[str]:
    """Names of the depths `context_states` returns, in forward order:
    "embed" (encoder input), then "block<l>/sample" and "block<l>/node"
    (after the sample- and node-attention layer of encoder block l)."""
    stack = model.encoder.encoder
    names = ["embed"]
    for i in range(len(stack.node_layers)):
        names += [f"block{i}/sample", f"block{i}/node"]
    return names


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
) -> dict[str, torch.Tensor]:
    """Residual-stream representation of every CONTEXT cell at each depth of
    `probe_points(model)`, for one context and one question (T, Y).

    Returns {name: float32 CPU tensor [N, D, d_model]}. Context rows attend
    only to context rows, so the states do not depend on the query do-values
    (a single dummy query point is used). The role embedding does depend on
    (T, Y), hence the arguments. zero_flags sets every intervened-cell flag to
    False, as in `predict_mixture`. Works for every mixed-context model class
    (the hooks sit on the upstream attention layers, which all share).
    Internal node positions beyond the D variables (e.g. the extra-node
    metadata column) are dropped.
    """
    stack = model.encoder.encoder
    Nc = int(np.asarray(ctx_x).shape[0])
    D = int(np.asarray(ctx_x).shape[1])
    B = 1
    states: dict[str, torch.Tensor] = {}

    def keep(name, x):  # x [B, Nc, D', d_model]; D' = D + internal positions
        states[name] = x.detach().float().cpu()[0, :, :D]

    def pre_embed(_mod, args):
        keep("embed", args[0][:, :Nc])

    def after_sample(name, ctx_only):
        def hook(_mod, _args, out):  # out [B * D, S, d_model]
            S = out.shape[1]
            x = out.reshape(B, -1, S, out.shape[-1]).permute(0, 2, 1, 3)
            keep(name, x if ctx_only else x[:, :Nc])

        return hook

    def after_node(name):
        def hook(_mod, _args, out):  # out [B * S, D, d_model]
            x = out.reshape(B, -1, *out.shape[1:])
            keep(name, x[:, :Nc])

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
