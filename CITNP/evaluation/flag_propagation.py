"""Single-flag perturbation diagnostic: how removing context intervention
flags changes the encoder's hidden states and the model's predictions.

For one context with intervention mask M, a perturbed mask M- removes flags
(1 -> 0) and nothing else; observed values, queries and the per-task
normalisation (computed once from the ORIGINAL mask) are unchanged. Every
recorded token is compared between the two forward passes:

    delta_h_t = h_t(X, M) - h_t(X, M-)        at every stage t

Two ablation modes:
- "single_flag": one source cell (r, k) with M[r, k] = 1;
- "regime": every flag of variable k in the rows of one regime g. On the
  current banks every interventional regime has a single target, so this
  turns the whole regime into rows that look observational; read it as
  "regime relabelled as observational".

The measure is computational dependence of the forward pass on the flags. It
is not information content, intervention understanding or causal correctness,
and a change at a destination does not identify the head or path that
carried it.

Stages and hook location
------------------------
Stages are `model_api.probe_points(model)`: "embed", then "block<l>/sample"
and "block<l>/node". A stage is the residual stream at the OUTPUT of the
pre-norm sample- or node-attention layer of block l (after both residual
additions, attention and feed-forward), before any later layer-wise
re-injection. "embed" is the input of the transformer stack, including scale
conditioning. With MHCA, context rows are read from the block's
self-attention layer and query rows from its cross-attention layer. See
`model_api._record_states`.

Token layout
------------
Context tokens are [N, D'] with D' = D (layerwise) or D + 1 (extra-node, the
row's metadata token at position D). Query tokens are the n_t query rows of
the bank item (one per do-value), [n_t, D'] likewise; the extra-node query
metadata token is kept as its own category ("query_metadata").

Layer-wise injection decomposition
----------------------------------
Layer-wise models add the masked flag code `mask * MLP(x)` to the residual
stream at several sites (`layerwise_injection_sites`). The FIRST injection is
every flag entry before the first attention operation: the input embedding
(`context_intervened`) plus `inject_sample[0]`. LATER injections are
`inject_sample[i]` for i >= 1 and every `inject_node[i]`. Note that the
"embed" stage is recorded before `inject_sample[0]`, so it captures only part
of the first injection. Injection modes:
- "all_injections": the perturbed mask everywhere (plain forward pass);
- "first_injection_only": perturbed mask at the model input, the original
  mask's code at the later sites;
- "later_injections_only": original mask at the model input, the perturbed
  mask's code at the later sites.
The code is mask * MLP(x) and x does not depend on the mask, so the code
under any mask M' contained in M is exactly code(M) * M'. Forward-pre-hooks
on the later sites rebuild it from batch item 0, which is therefore always
the original-mask baseline. Extra-node models have no re-injection.

Regimes
-------
A bank item's `ctx_regime` (row -> regime index) is used when present;
otherwise rows with identical intervention-target sets form one regime (two
regimes with the same target set would be merged; on the banks of
`tasks.build_bank` every interventional regime has a distinct single target,
so the grouping is exact). A regime is observational if none of its rows has
a flag.

Destination categories (mean per destination token, missing = NaN)
---------------------------------------------------------------
With R_s the source rows, k the source variable and R_g the source's regime:
source_cells (i in R_s, j = k); source_rows_other_variables (i in R_s, j != k);
regime_mates_same_variable (i in R_g \\ R_s, j = k);
other_interventional_rows_same_variable; observational_rows_same_variable;
other_rows_other_variables (i not in R_s, j != k); for extra-node
metadata_source_rows / metadata_other_rows; query_cells and query_metadata.
The first six partition the valid ordinary context cells.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from CITNP.evaluation.model_api import (
    _mixture_on_raw_scale,
    _model_inputs,
    _record_states,
    hidden_state_layout,
    mixture_logpdf,
    probe_points,
)

SUPPORTED_ARCHITECTURES = ("layerwise", "extra_node")
ABLATION_MODES = ("single_flag", "regime")
INJECTION_MODES = (
    "all_injections",
    "first_injection_only",
    "later_injections_only",
)
ORDINARY_CATEGORIES = (
    "source_cells",
    "source_rows_other_variables",
    "regime_mates_same_variable",
    "other_interventional_rows_same_variable",
    "observational_rows_same_variable",
    "other_rows_other_variables",
)
METADATA_CATEGORIES = ("metadata_source_rows", "metadata_other_rows")
QUERY_CATEGORIES = ("query_cells", "query_metadata")

# Relative and normalised metrics are NaN when their denominator is below this.
DENOMINATOR_TOLERANCE = 1e-6
# Added to denominators of rel_l2 and source_normalised_sensitivity.
RELATIVE_EPSILON = 1e-8
DEFAULT_KL_SAMPLES = 1000


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def check_supported_model(model: torch.nn.Module) -> dict:
    """`hidden_state_layout(model)`, or ValueError for unsupported ones."""
    layout = hidden_state_layout(model)
    if layout["architecture"] not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"flag-propagation supports {SUPPORTED_ARCHITECTURES}; this model "
            f"is {layout['architecture']!r}"
        )
    return layout


def check_variable_count(model: torch.nn.Module, num_variables: int) -> None:
    """Raise if a context with `num_variables` variables cannot be run."""
    if num_variables > model.num_nodes:
        raise ValueError(
            f"context has {num_variables} variables but the model was built "
            f"for num_nodes={model.num_nodes} (see --num_nodes)"
        )


def layerwise_injection_sites(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module | None, str]]:
    """Every point where a layer-wise model adds the flag code to the
    residual stream, in forward order, as (name, module, group).

    group is "first" for flag entries before the first attention operation
    (the input embedding, which has no hookable module of its own, and
    `inject_sample[0]`) and "later" for entries after token mixing has
    started (`inject_sample[i]`, i >= 1, and every `inject_node[i]`). The
    decomposition separates flag information that entered before any mixing
    between tokens from information added after mixing started.
    """
    stack = model.encoder.encoder
    sites: list[tuple[str, torch.nn.Module | None, str]] = [
        ("embedding", None, "first")
    ]
    for i, sample_map in enumerate(stack.inject_sample):
        sites.append((f"inject_sample[{i}]", sample_map, "first" if i == 0 else "later"))  # fmt: skip
        if i < len(stack.inject_node):
            sites.append((f"inject_node[{i}]", stack.inject_node[i], "later"))
    return sites


# --------------------------------------------------------------------------- #
# Regimes and sources
# --------------------------------------------------------------------------- #
def regime_labels(item: dict) -> tuple[np.ndarray, str]:
    """(regime index per context row [N], how it was obtained).

    "ctx_regime" if the item stores it, else "target_set": rows with the
    same set of intervened variables share a regime, numbered in sorted
    order of the target sets (so the observational set () is 0 if present).
    """
    if item.get("ctx_regime") is not None:
        labels = np.asarray(item["ctx_regime"], dtype=np.int64)
        if labels.shape != (np.asarray(item["ctx_int"]).shape[0],):
            raise ValueError("ctx_regime must have one entry per context row")
        return labels, "ctx_regime"
    mask = np.asarray(item["ctx_int"], dtype=bool)
    keys = [tuple(np.flatnonzero(row).tolist()) for row in mask]
    order = {k: g for g, k in enumerate(sorted(set(keys)))}
    return np.array([order[k] for k in keys], dtype=np.int64), "target_set"


def observational_rows(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """bool [N]: the row belongs to a regime in which no row has a flag."""
    mask = np.asarray(mask, dtype=bool)
    flagged = {int(g) for g in np.unique(labels[mask.any(1)])}
    return np.array([int(g) not in flagged for g in labels])


@dataclass(frozen=True)
class FlagSource:
    """Flags to remove: variable `variable` in rows `rows` (all in regime
    `regime`)."""

    source_id: int
    ablation_mode: str
    variable: int
    rows: tuple[int, ...]
    regime: int

    @property
    def cells(self) -> list[tuple[int, int]]:
        return [(r, self.variable) for r in self.rows]


def _round_robin(groups: dict[int, list], rng) -> list:
    """Interleave the (already shuffled) lists of `groups`, visiting the
    keys in random order, so consecutive picks vary the key."""
    keys = [int(k) for k in rng.permutation(sorted(groups))]
    out, depth = [], 0
    while any(depth < len(groups[k]) for k in keys):
        out += [groups[k][depth] for k in keys if depth < len(groups[k])]
        depth += 1
    return out


def select_single_flag_sources(
    context_int_mask: np.ndarray, labels: np.ndarray, *, max_sources: int, rng
) -> list[FlagSource]:
    """Up to `max_sources` distinct flagged cells, spread over the
    intervened variables (round robin over variables, random rows)."""
    mask = np.asarray(context_int_mask, dtype=bool)
    groups = {
        int(k): [int(r) for r in rng.permutation(np.flatnonzero(mask[:, k]))]
        for k in range(mask.shape[1])
        if mask[:, k].any()
    }
    order = _round_robin({k: [(r, k) for r in groups[k]] for k in groups}, rng)
    return [
        FlagSource(i, "single_flag", k, (r,), int(labels[r]))
        for i, (r, k) in enumerate(order[:max_sources])
    ]


def select_regime_sources(
    context_int_mask: np.ndarray,
    labels: np.ndarray,
    *,
    max_sources: int,
    rng,
    first_id: int = 0,
) -> list[FlagSource]:
    """Up to `max_sources` (regime, variable) pairs with at least one flag;
    each removes every flag of that variable in that regime. Varies the
    variable first."""
    mask = np.asarray(context_int_mask, dtype=bool)
    groups: dict[int, list] = {}
    for g in np.unique(labels):
        rows = np.flatnonzero(labels == g)
        for k in np.flatnonzero(mask[rows].any(0)):
            flagged = tuple(int(r) for r in rows[mask[rows, k]])
            groups.setdefault(int(k), []).append((int(g), int(k), flagged))
    groups = {k: [v[i] for i in rng.permutation(len(v))] for k, v in groups.items()}
    order = _round_robin(groups, rng)
    return [
        FlagSource(first_id + i, "regime", k, rows, g)
        for i, (g, k, rows) in enumerate(order[:max_sources])
    ]


def remove_flags(context_int_mask, cells) -> torch.Tensor | np.ndarray:
    """Copy of the [N, D] mask with the flags of `cells` ((row, variable)
    pairs) set to 0. Raises if a cell is not flagged: flags are only ever
    removed."""
    out = (
        context_int_mask.clone()
        if isinstance(context_int_mask, torch.Tensor)
        else np.array(context_int_mask, dtype=bool, copy=True)
    )
    for r, k in cells:
        if not bool(out[r, k]):
            raise ValueError(f"cell ({r}, {k}) has no flag to remove")
        out[r, k] = False
    return out


# --------------------------------------------------------------------------- #
# Batched forward passes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One perturbed batch item: which mask the model reads at its input and
    which mask the later layer-wise injection sites use."""

    source: FlagSource
    injection_mode: str
    input_mask: torch.Tensor  # [N, D] bool
    later_injection_mask: torch.Tensor  # [N, D] bool

    @property
    def input_mask_kind(self) -> str:
        return "original" if self.injection_mode == "later_injections_only" else "perturbed"  # fmt: skip

    @property
    def later_injection_mask_kind(self) -> str:
        return "original" if self.injection_mode == "first_injection_only" else "perturbed"  # fmt: skip


def make_variants(
    original_mask: torch.Tensor, source: FlagSource, injection_modes
) -> list[Variant]:
    perturbed = remove_flags(original_mask, source.cells)
    masks = {
        "all_injections": (perturbed, perturbed),
        "first_injection_only": (perturbed, original_mask),
        "later_injections_only": (original_mask, perturbed),
    }
    return [Variant(source, m, *masks[m]) for m in injection_modes]


def build_batch_inputs(prepared: dict, input_masks: list[torch.Tensor]) -> dict:
    """Repeat the single-item prepared inputs (`_model_inputs` with Q = 1)
    once per mask; only `context_int_mask` differs between items."""
    B = len(input_masks)
    batch = {}
    for key, value in prepared.items():
        if value.shape[0] != 1:
            raise ValueError(f"prepared input {key!r} must have batch size 1")
        batch[key] = value.repeat(B, *[1] * (value.ndim - 1))
    batch["context_int_mask"] = torch.stack(input_masks)
    return batch


@contextmanager
def later_injection_override(
    model: torch.nn.Module,
    input_masks: torch.Tensor,
    later_injection_masks: torch.Tensor,
) -> Iterator[None]:
    """Inside the block, every later injection site (see
    `layerwise_injection_sites`) of a layer-wise model receives, for each
    batch item b whose later-injection mask differs from its input mask,
    code[0] * later_injection_masks[b] instead of its own code.

    Item 0 must be the original-mask baseline (equal input and later masks)
    and every later mask must lie inside it. Hooks are removed on exit,
    also after an exception.
    """
    if not torch.equal(input_masks[0], later_injection_masks[0]):
        raise ValueError("batch item 0 must be the original-mask baseline")
    if (later_injection_masks & ~input_masks[0]).any():
        raise ValueError("later-injection masks may only remove flags")
    replace = torch.nonzero(
        (input_masks != later_injection_masks).flatten(1).any(1)
    ).flatten()
    later = later_injection_masks[replace]  # [R, N, D]

    def hook(_module, args):
        code = args[0]  # [B, Nc + Nq, D, d_model], zero on query rows
        if len(replace) == 0:
            return None
        m = later.to(code.device, code.dtype)[..., None]
        pad = m.new_zeros(len(replace), code.shape[1] - m.shape[1], *m.shape[2:])
        new = code.clone()
        new[replace.to(code.device)] = code[:1] * torch.cat([m, pad], dim=1)
        return (new, *args[1:])

    handles = []
    try:
        for _name, module, group in layerwise_injection_sites(model):
            if group == "later":
                handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def compute_state_differences(
    original: torch.Tensor, perturbed: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-token L2 norms (float64, CPU) of h(M) - h(M-) and of h(M).

    original [..., d] broadcasts against perturbed [B, ..., d]; returns
    abs_l2 [B, ...] and original_l2 [...]."""
    o, p = original.double(), perturbed.double()
    return {
        "abs_l2": (o - p).norm(dim=-1).cpu(),
        "original_l2": o.norm(dim=-1).cpu(),
    }


def _differences_to_item0(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """x [B, rows, D', d] -> abs_l2 [B, rows, D'] and original_l2 [rows, D']
    against batch item 0 (the original-mask baseline)."""
    return compute_state_differences(x[0], x)


@torch.inference_mode()
def run_batch(
    model: torch.nn.Module,
    prepared: dict,
    variants: list[Variant],
    original_mask: torch.Tensor,
    *,
    device,
    layout: dict,
):
    """One forward pass: item 0 is the original-mask baseline, items 1.. are
    `variants`. Returns ({stage: {"context"/"query": per-token norms}},
    model output)."""
    input_masks = [original_mask] + [v.input_mask for v in variants]
    later_masks = [original_mask] + [v.later_injection_mask for v in variants]
    batch = build_batch_inputs(prepared, input_masks)
    needs_override = any(
        v.injection_mode != "all_injections" for v in variants
    )
    if needs_override and layout["architecture"] != "layerwise":
        raise ValueError("injection decomposition needs a layer-wise model")
    if needs_override:
        with later_injection_override(
            model, torch.stack(input_masks), torch.stack(later_masks)
        ):
            return _record_states(model, batch, device, keep=_differences_to_item0)
    return _record_states(model, batch, device, keep=_differences_to_item0)


# --------------------------------------------------------------------------- #
# Destination categories and per-category metrics
# --------------------------------------------------------------------------- #
def context_category_masks(
    source: FlagSource,
    labels: np.ndarray,
    is_observational: np.ndarray,
    *,
    num_variables: int,
    valid_rows: np.ndarray | None = None,
    valid_variables: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """bool [N, num_variables] per ordinary category (valid cells only),
    plus bool [N] per metadata category (valid rows only)."""
    N = len(labels)
    vr = np.ones(N, bool) if valid_rows is None else np.asarray(valid_rows, bool)
    vv = (
        np.ones(num_variables, bool)
        if valid_variables is None
        else np.asarray(valid_variables, bool)
    )
    in_source = np.zeros(N, bool)
    in_source[list(source.rows)] = True
    same_regime = labels == source.regime
    same_var = np.zeros(num_variables, bool)
    same_var[source.variable] = True
    other_interventional = ~same_regime & ~is_observational
    valid = vr[:, None] & vv[None, :]

    def cells(rows, cols):
        return rows[:, None] & cols[None, :] & valid

    masks = {
        "source_cells": cells(in_source, same_var),
        "source_rows_other_variables": cells(in_source, ~same_var),
        "regime_mates_same_variable": cells(same_regime & ~in_source, same_var),
        "other_interventional_rows_same_variable": cells(
            other_interventional & ~in_source, same_var
        ),
        "observational_rows_same_variable": cells(
            is_observational & ~same_regime & ~in_source, same_var
        ),
        "other_rows_other_variables": cells(~in_source, ~same_var),
        "metadata_source_rows": in_source & vr,
        "metadata_other_rows": ~in_source & vr,
    }
    return masks


def stage_reference_l2(original_l2: np.ndarray, valid_cells: np.ndarray) -> float:
    """Median baseline token norm over the valid ordinary context cells."""
    values = np.asarray(original_l2)[np.asarray(valid_cells, bool)]
    return float(np.median(values)) if values.size else float("nan")


def summarise_tokens(
    abs_l2: np.ndarray,
    original_l2: np.ndarray,
    selected: np.ndarray,
    *,
    stage_median: float,
    d_model: int,
) -> dict[str, float]:
    """Mean per destination token of each metric over `selected` tokens.
    Empty selections and undefined ratios give NaN (never 0 or inf)."""
    a = np.asarray(abs_l2, np.float64)[selected]
    o = np.asarray(original_l2, np.float64)[selected]
    n = int(a.size)
    if n == 0:
        nan = float("nan")
        return dict(mean_abs_l2=nan, mean_rms=nan, mean_rel_l2=nan,
                    mean_stage_normalised_l2=nan, max_abs_l2=nan,
                    num_destinations=0)  # fmt: skip
    rel = np.where(o >= DENOMINATOR_TOLERANCE, a / (o + RELATIVE_EPSILON), np.nan)
    stage_ok = np.isfinite(stage_median) and stage_median >= DENOMINATOR_TOLERANCE
    return dict(
        mean_abs_l2=float(a.mean()),
        mean_rms=float(a.mean() / np.sqrt(d_model)),
        mean_rel_l2=float(rel[np.isfinite(rel)].mean()) if np.isfinite(rel).any() else float("nan"),  # fmt: skip
        mean_stage_normalised_l2=float(a.mean() / stage_median) if stage_ok else float("nan"),  # fmt: skip
        max_abs_l2=float(a.max()),
        num_destinations=n,
    )


def aggregate_by_category(
    context_abs_l2: np.ndarray,
    context_original_l2: np.ndarray,
    query_abs_l2: np.ndarray,
    query_original_l2: np.ndarray,
    source: FlagSource,
    labels: np.ndarray,
    is_observational: np.ndarray,
    *,
    num_variables: int,
    has_metadata: bool,
    d_model: int,
    stage_median: float,
    valid_rows: np.ndarray | None = None,
    valid_variables: np.ndarray | None = None,
    valid_queries: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Per-category metrics at one stage for one perturbation.

    context arrays [N, D'] and query arrays [n_q, D'] are per-token norms
    (D' = num_variables, + 1 metadata column at index num_variables when
    `has_metadata`). `stage_median` comes from the baseline pass
    (`stage_reference_l2`)."""
    masks = context_category_masks(
        source, labels, is_observational, num_variables=num_variables,
        valid_rows=valid_rows, valid_variables=valid_variables,
    )  # fmt: skip
    kw = dict(stage_median=stage_median, d_model=d_model)
    D = num_variables
    out = {
        c: summarise_tokens(context_abs_l2[:, :D], context_original_l2[:, :D], masks[c], **kw)  # fmt: skip
        for c in ORDINARY_CATEGORIES
    }
    if has_metadata:
        for c in METADATA_CATEGORIES:
            out[c] = summarise_tokens(context_abs_l2[:, D], context_original_l2[:, D], masks[c], **kw)  # fmt: skip
    nq = query_abs_l2.shape[0]
    vq = np.ones(nq, bool) if valid_queries is None else np.asarray(valid_queries, bool)
    vv = np.ones(D, bool) if valid_variables is None else np.asarray(valid_variables, bool)
    out["query_cells"] = summarise_tokens(
        query_abs_l2[:, :D], query_original_l2[:, :D], vq[:, None] & vv[None, :], **kw
    )
    if has_metadata:
        out["query_metadata"] = summarise_tokens(
            query_abs_l2[:, D], query_original_l2[:, D], vq, **kw
        )
    return out


def initial_source_magnitude(
    embed_abs_l2: np.ndarray, source: FlagSource, *, num_variables: int,
    has_metadata: bool,
) -> float:  # fmt: skip
    """Mean abs_l2 at "embed" of the tokens the removed flags enter: the
    source cells (layer-wise) or the source rows' metadata tokens
    (extra-node)."""
    rows = list(source.rows)
    if has_metadata:
        return float(np.mean(embed_abs_l2[rows, num_variables]))
    return float(np.mean(embed_abs_l2[rows, source.variable]))


def source_normalised(category_mean: float, initial: float) -> float:
    """category mean / initial source magnitude, or NaN when the initial
    magnitude is at most DENOMINATOR_TOLERANCE. Not a conserved fraction."""
    if not np.isfinite(initial) or initial <= DENOMINATOR_TOLERANCE:
        return float("nan")
    return float(category_mean / (initial + RELATIVE_EPSILON))


# --------------------------------------------------------------------------- #
# Output-level change
# --------------------------------------------------------------------------- #
def _mixture_moments(mu, sd, logw):
    w = logw.exp()
    mean = (w * mu).sum(-1)
    var = (w * (sd**2 + mu**2)).sum(-1) - mean**2
    return mean, var.clamp_min(0).sqrt()


def compute_predictive_changes(
    original: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    perturbed: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_samples: int = DEFAULT_KL_SAMPLES,
    seed: int = 0,
) -> dict[str, np.ndarray | str]:
    """KL(p(.|M) || p(.|M-)) and moment shifts per query point.

    original: (mu, sd, logw) [n_t, K]; perturbed: the same with a leading
    axis [P, n_t, K] (raw data scale, float64). With K = 1 the Gaussian KL
    is exact ("closed_form", standard error 0). Otherwise it is a Monte
    Carlo estimate ("monte_carlo"): `num_samples` draws per query point from
    the original-mask predictive, seeded by `seed`, the same draws for every
    perturbed item; the estimate is not clipped, so it can be slightly
    negative, and its standard error is returned.

    Returns predictive_kl, kl_standard_error, mean_shift, std_shift (each
    [P, n_t] numpy) and kl_method.
    """
    mu0, sd0, lw0 = (p.detach() for p in original)
    mu1, sd1, lw1 = (p.detach() for p in perturbed)
    K = mu0.shape[-1]
    m0, s0 = _mixture_moments(mu0, sd0, lw0)
    m1, s1 = _mixture_moments(mu1, sd1, lw1)
    if K == 1:
        a, b = sd0[..., 0], sd1[..., 0]
        kl = torch.log(b / a) + (a**2 + (mu0[..., 0] - mu1[..., 0]) ** 2) / (2 * b**2) - 0.5  # fmt: skip
        se = torch.zeros_like(kl)
        method = "closed_form"
    else:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        comp = torch.multinomial(lw0.exp().cpu(), num_samples, replacement=True, generator=g)  # fmt: skip
        z = torch.randn(comp.shape, generator=g, dtype=torch.float64)
        comp = comp.to(mu0.device)
        y = torch.gather(mu0, -1, comp) + torch.gather(sd0, -1, comp) * z.to(mu0.device)  # fmt: skip
        lp0 = mixture_logpdf(mu0[:, None], sd0[:, None], lw0[:, None], y)
        lp1 = mixture_logpdf(mu1[:, :, None], sd1[:, :, None], lw1[:, :, None], y)
        diff = lp0 - lp1  # [P, n_t, S]
        kl = diff.mean(-1)
        se = diff.std(-1, unbiased=True) / np.sqrt(num_samples)
        method = "monte_carlo"
    return dict(
        predictive_kl=kl.cpu().numpy(),
        kl_standard_error=se.cpu().numpy(),
        mean_shift=(m0 - m1).abs().cpu().numpy(),
        std_shift=(s0 - s1).abs().cpu().numpy(),
        kl_method=method,
    )


# --------------------------------------------------------------------------- #
# One context
# --------------------------------------------------------------------------- #
def _source_fields(source: FlagSource, mask: np.ndarray, labels: np.ndarray) -> dict:
    perturbed = remove_flags(mask, source.cells)
    rows = list(source.rows)
    return dict(
        source_id=source.source_id,
        ablation_mode=source.ablation_mode,
        source_rows=";".join(str(r) for r in rows),
        source_variable=source.variable,
        source_regime=source.regime,
        source_regime_size=int((labels == source.regime).sum()),
        num_source_cells=len(rows),
        row_becomes_observational=bool((~perturbed[rows].any(1)).any()),
        source_is_multi_target=bool((mask[rows].sum(1) > 1).any()),
    )


def analyse_context(
    model: torch.nn.Module,
    item: dict,
    sources: list[FlagSource],
    *,
    device,
    normalise: bool = True,
    injection_decomposition: bool = True,
    max_batch_size: int = 64,
    kl_samples: int = DEFAULT_KL_SAMPLES,
    kl_seed: int = 0,
    context_id: int = 0,
    keep_token_maps: bool = False,
) -> dict:
    """Run the baseline and every perturbation of `sources` on one bank
    item and aggregate.

    Returns dict with
    - summary: list of row dicts (one per variant / stage / category);
    - predictions: list of row dicts (one per variant / query point);
    - token_maps (if keep_token_maps): {(source_id, injection_mode):
      {"context_abs_l2": [stages, N, D'], "context_original_l2": ...,
      "query_abs_l2": [stages, n_q, D'], "query_original_l2": ...}}.
    """
    layout = check_supported_model(model)
    mask_np = np.asarray(item["ctx_int"], dtype=bool)
    N, D = mask_np.shape
    check_variable_count(model, D)
    labels, _ = regime_labels(item)
    is_obs = observational_rows(mask_np, labels)
    t = np.asarray(item.get("t", np.zeros(1)), dtype=np.float64).ravel()
    T, Y = int(item["T"]), int(item["Y"])
    prepared, mean, std = _model_inputs(
        item["ctx_x"], mask_np, T, Y, t, zero_flags=False, normalise=normalise
    )
    original_mask = prepared["context_int_mask"][0]
    layerwise = layout["architecture"] == "layerwise"
    modes = INJECTION_MODES if (layerwise and injection_decomposition) else ("all_injections",)  # fmt: skip
    variants = [v for s in sources for v in make_variants(original_mask, s, modes)]
    stages = probe_points(model)
    has_meta = layout["metadata_token"]
    d_model = int(model.d_model)
    per_chunk = max(1, int(max_batch_size) - 1)

    summary, predictions, token_maps = [], [], {}
    for start in range(0, len(variants), per_chunk):
        chunk = variants[start : start + per_chunk]
        states, out = run_batch(
            model, prepared, chunk, original_mask, device=device, layout=layout
        )
        mu, sd, logw = _mixture_on_raw_scale(
            out, prepared["outcome_index"].repeat(len(chunk) + 1), mean, std, device
        )
        changes = compute_predictive_changes(
            (mu[0], sd[0], logw[0]), (mu[1:], sd[1:], logw[1:]),
            num_samples=kl_samples, seed=kl_seed,
        )  # fmt: skip
        medians = {
            s: stage_reference_l2(states[s]["context"]["original_l2"][:, :D], np.ones((N, D), bool))  # fmt: skip
            for s in stages
        }
        for b, v in enumerate(chunk, start=1):
            fields = dict(
                context_id=context_id,
                **_source_fields(v.source, mask_np, labels),
                injection_mode=v.injection_mode,
                input_mask=v.input_mask_kind,
                later_injection_mask=v.later_injection_mask_kind if layerwise else "none",  # fmt: skip
                query_treatment=T,
                query_outcome=Y,
            )
            initial = initial_source_magnitude(
                states["embed"]["context"]["abs_l2"][b].numpy(), v.source,
                num_variables=D, has_metadata=has_meta,
            )  # fmt: skip
            for si, s in enumerate(stages):
                ctx, qry = states[s]["context"], states[s]["query"]
                cats = aggregate_by_category(
                    ctx["abs_l2"][b].numpy(), ctx["original_l2"].numpy(),
                    qry["abs_l2"][b].numpy(), qry["original_l2"].numpy(),
                    v.source, labels, is_obs, num_variables=D,
                    has_metadata=has_meta, d_model=d_model,
                    stage_median=medians[s],
                )  # fmt: skip
                for cat, metrics in cats.items():
                    summary.append(dict(
                        fields, stage_name=s, stage_index=si, category=cat,
                        **metrics, initial_source_l2=initial,
                        source_normalised_sensitivity=source_normalised(metrics["mean_abs_l2"], initial),
                    ))  # fmt: skip
            for q in range(len(t)):
                predictions.append(dict(
                    fields, query_index=q, query_do_value=float(t[q]),
                    predictive_kl=float(changes["predictive_kl"][b - 1, q]),
                    kl_standard_error=float(changes["kl_standard_error"][b - 1, q]),
                    mean_shift=float(changes["mean_shift"][b - 1, q]),
                    std_shift=float(changes["std_shift"][b - 1, q]),
                    kl_method=changes["kl_method"],
                ))  # fmt: skip
            if keep_token_maps:
                token_maps[(v.source.source_id, v.injection_mode)] = {
                    f"{part}_{key}": np.stack([
                        states[s][part][key][b].numpy() if key == "abs_l2"
                        else states[s][part][key].numpy()
                        for s in stages
                    ])
                    for part in ("context", "query")
                    for key in ("abs_l2", "original_l2")
                }  # fmt: skip
    return dict(summary=summary, predictions=predictions, token_maps=token_maps)


# --------------------------------------------------------------------------- #
# Whole bank
# --------------------------------------------------------------------------- #
def select_contexts(bank: list[dict], max_contexts: int, seed: int) -> list[int]:
    """Bank indices of up to `max_contexts` distinct contexts (first item of
    each) that have at least one context intervention flag, shuffled with
    `seed`."""
    first: dict[bytes, int] = {}
    for i, it in enumerate(bank):
        if np.asarray(it["ctx_int"], dtype=bool).any():
            first.setdefault(np.ascontiguousarray(it["ctx_x"]).tobytes(), i)
    idx = list(first.values())
    order = np.random.default_rng(seed).permutation(len(idx))
    return [idx[i] for i in order[:max_contexts]]


def select_sources(
    item: dict, *, sources_per_context: int, regime_sources_per_context: int,
    ablation: str, seed: int, context_id: int,
) -> list[FlagSource]:  # fmt: skip
    """Single-flag then regime sources of one context, reproducible from
    (seed, context_id) alone."""
    mask = np.asarray(item["ctx_int"], dtype=bool)
    labels, _ = regime_labels(item)
    rng = np.random.default_rng([seed, context_id])
    out: list[FlagSource] = []
    if ablation in ("both", "single_flag"):
        out += select_single_flag_sources(mask, labels, max_sources=sources_per_context, rng=rng)  # fmt: skip
    if ablation in ("both", "regime"):
        out += select_regime_sources(
            mask, labels, max_sources=regime_sources_per_context, rng=rng,
            first_id=len(out),
        )  # fmt: skip
    return out


def run_flag_propagation(
    model: torch.nn.Module,
    bank: list[dict],
    *,
    device,
    normalise: bool = True,
    max_contexts: int = 100,
    sources_per_context: int = 3,
    regime_sources_per_context: int = 1,
    ablation: str = "both",
    injection_decomposition: bool = True,
    max_batch_size: int = 64,
    kl_samples: int = DEFAULT_KL_SAMPLES,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Flag-propagation diagnostic on up to `max_contexts` bank contexts.

    Returns {"summary": per variant / stage / category rows,
    "predictions": per variant / query point rows}. The model must be in
    eval mode; its weights are never changed.
    """
    if ablation not in ("both", *ABLATION_MODES):
        raise ValueError(f"ablation must be 'both' or one of {ABLATION_MODES}")
    check_supported_model(model)
    summary, predictions = [], []
    for context_id in select_contexts(bank, max_contexts, seed):
        item = bank[context_id]
        sources = select_sources(
            item, sources_per_context=sources_per_context,
            regime_sources_per_context=regime_sources_per_context,
            ablation=ablation, seed=seed, context_id=context_id,
        )  # fmt: skip
        res = analyse_context(
            model, item, sources, device=device, normalise=normalise,
            injection_decomposition=injection_decomposition,
            max_batch_size=max_batch_size, kl_samples=kl_samples,
            kl_seed=int(np.random.SeedSequence([seed, context_id]).generate_state(1)[0]),
            context_id=context_id,
        )  # fmt: skip
        summary += res["summary"]
        predictions += res["predictions"]
    return dict(summary=pd.DataFrame(summary), predictions=pd.DataFrame(predictions))
