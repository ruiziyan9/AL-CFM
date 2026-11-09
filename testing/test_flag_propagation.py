"""Tests for CITNP/evaluation/flag_propagation.py on tiny random real
encoders (layer-wise shared / separate flag MLP, extra-node), CPU, float64."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flag_propagation_spec import D, bank_items, build_model, regime_context

from CITNP.evaluation.flag_propagation import (
    INJECTION_MODES,
    ORDINARY_CATEGORIES,
    FlagSource,
    Variant,
    aggregate_by_category,
    analyse_context,
    build_batch_inputs,
    build_source_manifest,
    check_supported_model,
    compute_predictive_changes,
    compute_state_differences,
    context_category_masks,
    layerwise_injection_sites,
    later_injection_override,
    make_variants,
    observational_rows,
    regime_labels,
    remove_flags,
    run_batch,
    run_flag_propagation,
    select_regime_sources,
    select_single_flag_sources,
    select_sources,
    source_normalised,
    sources_from_manifest,
    stage_reference_l2,
    summarise_tokens,
)
from CITNP.evaluation.model_api import (
    _call,
    _mixture_on_raw_scale,
    _model_inputs,
    _record_states,
    context_states,
    probe_points,
)

LAYERWISE = ["layerwise_shared", "layerwise_separate"]
ALL_ARCHS = [*LAYERWISE, "extra_node"]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    return {a: build_model(tmp, a) for a in ALL_ARCHS}


@pytest.fixture(scope="module")
def item():
    x, ci, regime = regime_context(seed=1, n_obs=10, sizes=(3, 2), targets=(0, 2))
    return dict(ctx_x=x, ctx_int=ci, ctx_regime=regime, T=1, Y=2,
                t=np.array([-1.0, 0.5], dtype=np.float32))  # fmt: skip


def prepare(item):
    prepared, mean, std = _model_inputs(
        item["ctx_x"], item["ctx_int"], item["T"], item["Y"], item["t"],
        zero_flags=False, normalise=True,
    )  # fmt: skip
    return prepared, mean, std, prepared["context_int_mask"][0]


def single_source(item, row, var):
    labels, _ = regime_labels(item)
    return FlagSource(0, "single_flag", var, (row,), int(labels[row]))


def hook_count(model):
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )


# 1. Flag removal ------------------------------------------------------------
def test_remove_flags_single_and_regime(item):
    mask = np.asarray(item["ctx_int"])
    before = mask.copy()
    out = remove_flags(mask, [(11, 0)])
    assert not out[11, 0] and (out ^ mask).sum() == 1
    rows = np.flatnonzero(item["ctx_regime"] == 1)
    out = remove_flags(mask, [(r, 0) for r in rows])
    assert (out ^ mask).sum() == len(rows) and not out[rows, 0].any()
    np.testing.assert_array_equal(mask, before)  # original unchanged
    t = torch.as_tensor(mask)
    out_t = remove_flags(t, [(11, 0)])
    assert t[11, 0] and not out_t[11, 0]
    with pytest.raises(ValueError, match="no flag"):
        remove_flags(mask, [(0, 0)])  # observational row


def test_regime_labels_and_observational_rows(item):
    labels, how = regime_labels(item)
    assert how == "ctx_regime"
    np.testing.assert_array_equal(labels, item["ctx_regime"])
    no_regime = {k: v for k, v in item.items() if k != "ctx_regime"}
    labels2, how2 = regime_labels(no_regime)
    assert how2 == "target_set"
    # same partition of rows (labels may be numbered differently)
    pairs = {(a, b) for a, b in zip(labels, labels2)}
    assert len(pairs) == len(set(labels)) == len(set(labels2))
    obs = observational_rows(item["ctx_int"], labels)
    np.testing.assert_array_equal(obs, item["ctx_regime"] == 0)


def test_source_selection(item):
    mask = np.asarray(item["ctx_int"])
    labels, _ = regime_labels(item)
    rng = np.random.default_rng(0)
    single = select_single_flag_sources(mask, labels, max_sources=4, rng=rng)
    cells = [c for s in single for c in s.cells]
    assert len(set(cells)) == 4 and all(mask[r, k] for r, k in cells)
    assert {s.variable for s in single[:2]} == {0, 2}  # varied targets first
    regime = select_regime_sources(mask, labels, max_sources=5, rng=rng, first_id=4)
    assert len(regime) == 2 and [s.source_id for s in regime] == [4, 5]
    for s in regime:
        np.testing.assert_array_equal(sorted(s.rows), np.flatnonzero(labels == s.regime))
    a = select_sources(item, sources_per_context=3, regime_sources_per_context=1,
                       ablation="both", seed=0, context_id=5)  # fmt: skip
    b = select_sources(item, sources_per_context=3, regime_sources_per_context=1,
                       ablation="both", seed=0, context_id=5)  # fmt: skip
    assert a == b and [s.ablation_mode for s in a] == ["single_flag"] * 3 + ["regime"]


# 2. Prepared-input equality -------------------------------------------------
@pytest.mark.parametrize("arch", LAYERWISE)
def test_inputs_differ_only_in_intended_mask_entries(models, item, arch):
    model = models[arch]
    prepared, _, _, M = prepare(item)
    source = single_source(item, 11, 0)
    variants = make_variants(M, source, INJECTION_MODES)
    perturbed = remove_flags(M, source.cells)
    seen = {}

    def spy(_mod, _args, kwargs):
        seen.update(kwargs)

    h = model.register_forward_pre_hook(spy, with_kwargs=True)
    try:
        run_batch(model, prepared, variants, M, device="cpu",
                  layout=check_supported_model(model))  # fmt: skip
    finally:
        h.remove()
    for key, value in prepared.items():
        if key == "context_int_mask":
            continue
        for b in range(len(variants) + 1):
            assert torch.equal(seen[key][b].cpu(), value[0].to(seen[key].dtype)), key
    masks = seen["context_int_mask"]
    assert torch.equal(masks[0], M)
    want_input = {"all_injections": perturbed, "first_injection_only": perturbed,
                  "later_injections_only": M}  # fmt: skip
    want_later = {"all_injections": perturbed, "first_injection_only": M,
                  "later_injections_only": perturbed}  # fmt: skip
    for b, v in enumerate(variants, start=1):
        assert torch.equal(masks[b], want_input[v.injection_mode])
        assert torch.equal(v.later_injection_mask, want_later[v.injection_mode])
        assert v.input_mask_kind == (
            "original" if v.injection_mode == "later_injections_only" else "perturbed"
        )


# 3. Determinism -------------------------------------------------------------
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_determinism(models, item, arch):
    sources = select_sources(item, sources_per_context=2, regime_sources_per_context=1,
                             ablation="both", seed=0, context_id=0)  # fmt: skip
    runs = [analyse_context(models[arch], item, sources, device="cpu") for _ in range(2)]  # fmt: skip
    for key in ("summary", "predictions"):  # DataFrame.equals: NaN == NaN
        assert pd.DataFrame(runs[0][key]).equals(pd.DataFrame(runs[1][key]))


# 4. No-change control / 12a. zero KL ------------------------------------------
# Items of one batch are not bitwise identical even with identical inputs:
# CPU BLAS kernels round differently by batch position (~1e-15 in float64).
# Exact zeros therefore hold between separate passes; within a batch the
# no-change control is zero up to this rounding.
BATCH_ROUNDING = 1e-12


@pytest.mark.parametrize("arch", ALL_ARCHS)
@torch.no_grad()
def test_identical_masks_give_exact_zero_between_passes(models, item, arch):
    model = models[arch]
    prepared, mean, std, M = prepare(item)
    batch = build_batch_inputs(prepared, [M])
    a, out_a = _record_states(model, batch, "cpu")
    b, out_b = _record_states(model, build_batch_inputs(prepared, [M.clone()]), "cpu")
    for s in a:
        for part in ("context", "query"):
            d = compute_state_differences(a[s][part][0], b[s][part])["abs_l2"]
            assert torch.count_nonzero(d) == 0, (s, part)
    pa = _mixture_on_raw_scale(out_a, prepared["outcome_index"], mean, std, "cpu")
    pb = _mixture_on_raw_scale(out_b, prepared["outcome_index"], mean, std, "cpu")
    ch = compute_predictive_changes(tuple(p[0] for p in pa), pb)
    assert ch["kl_method"] == "monte_carlo"
    for key in ("predictive_kl", "kl_standard_error", "mean_shift", "std_shift"):
        assert np.all(ch[key] == 0), key


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_identical_masks_in_a_batch_give_zero(models, item, arch):
    model = models[arch]
    layout = check_supported_model(model)
    prepared, mean, std, M = prepare(item)
    dummy = FlagSource(0, "single_flag", 0, (), 0)
    modes = INJECTION_MODES if layout["architecture"] == "layerwise" else ("all_injections",)  # fmt: skip
    variants = [Variant(dummy, m, M, M) for m in modes]
    states, out = run_batch(model, prepared, variants, M, device="cpu", layout=layout)
    for s in probe_points(model):
        for part in ("context", "query"):
            assert states[s][part]["abs_l2"].max() <= BATCH_ROUNDING, (s, part)
    mu, sd, logw = _mixture_on_raw_scale(
        out, prepared["outcome_index"].repeat(len(variants) + 1), mean, std, "cpu"
    )
    ch = compute_predictive_changes((mu[0], sd[0], logw[0]), (mu[1:], sd[1:], logw[1:]))  # fmt: skip
    for key in ("predictive_kl", "mean_shift", "std_shift"):
        assert np.abs(ch[key]).max() <= 1e-9, key


# 5. Batch equivalence -------------------------------------------------------
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_batched_baseline_equals_unbatched(models, item, arch):
    model = models[arch]
    prepared, _, _, M = prepare(item)
    perturbed = remove_flags(M, [(11, 0)])
    one, _ = _record_states(model, build_batch_inputs(prepared, [M]), "cpu")
    many, _ = _record_states(
        model, build_batch_inputs(prepared, [M, perturbed, perturbed, M]), "cpu"
    )
    for s in one:
        for part in ("context", "query"):
            torch.testing.assert_close(many[s][part][0], one[s][part][0], rtol=0, atol=1e-12)  # fmt: skip
            torch.testing.assert_close(many[s][part][3], one[s][part][0], rtol=0, atol=1e-12)  # fmt: skip
    # the public single-context wrapper agrees as well
    ctx = context_states(model, item["ctx_x"], item["ctx_int"], item["T"],
                         item["Y"], device="cpu", include_metadata_token=True)  # fmt: skip
    for s in one:
        torch.testing.assert_close(ctx[s].double(), one[s]["context"][0].float().double())
    # chunking does not change the results
    sources = select_sources(item, sources_per_context=3, regime_sources_per_context=1,
                             ablation="both", seed=0, context_id=0)  # fmt: skip
    small = analyse_context(model, item, sources, device="cpu", max_batch_size=3)
    large = analyse_context(model, item, sources, device="cpu", max_batch_size=64)
    for a, b in zip(small["summary"], large["summary"]):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], float):
                np.testing.assert_allclose(a[k], b[k], rtol=1e-9, atol=1e-12)
            else:
                assert a[k] == b[k]


# 6. / 7. Embedding locality -------------------------------------------------
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_embedding_locality(models, item, arch):
    model = models[arch]
    layout = check_supported_model(model)
    prepared, _, _, M = prepare(item)
    rows = np.flatnonzero(item["ctx_regime"] == 1)
    sources = [single_source(item, 11, 0), FlagSource(1, "regime", 0, tuple(rows), 1)]
    variants = [make_variants(M, s, ("all_injections",))[0] for s in sources]
    states, _ = run_batch(model, prepared, variants, M, device="cpu", layout=layout)
    embed = states["embed"]["context"]["abs_l2"]  # [B, N, D']
    for b, s in enumerate(sources, start=1):
        expected = torch.zeros_like(embed[b], dtype=torch.bool)
        if layout["metadata_token"]:
            expected[list(s.rows), D] = True  # source rows' metadata tokens
        else:
            expected[list(s.rows), s.variable] = True  # source cells
        assert embed[b][~expected].max() <= BATCH_ROUNDING
        assert (embed[b][expected] > 1e-6).all()
        assert states["embed"]["query"]["abs_l2"][b].max() <= BATCH_ROUNDING


# 8. Category partition ------------------------------------------------------
def _toy_layout():
    """N = 8, D = 3: observational rows 0-2, regime 1 = rows 3-5 on
    variable 0, regime 2 = rows 6-7 on variable 2."""
    mask = np.zeros((8, 3), bool)
    mask[3:6, 0] = True
    mask[6:8, 2] = True
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    return mask, labels


@pytest.mark.parametrize(
    "source, counts",
    [
        (FlagSource(0, "single_flag", 0, (4,), 1),
         dict(source_cells=1, source_rows_other_variables=2,
              regime_mates_same_variable=2,
              other_interventional_rows_same_variable=2,
              observational_rows_same_variable=3,
              other_rows_other_variables=14)),
        (FlagSource(0, "regime", 0, (3, 4, 5), 1),
         dict(source_cells=3, source_rows_other_variables=6,
              regime_mates_same_variable=0,
              other_interventional_rows_same_variable=2,
              observational_rows_same_variable=3,
              other_rows_other_variables=10)),
    ],
)  # fmt: skip
def test_category_partition(source, counts):
    mask, labels = _toy_layout()
    obs = observational_rows(mask, labels)
    masks = context_category_masks(source, labels, obs, num_variables=3)
    assert {c: int(masks[c].sum()) for c in ORDINARY_CATEGORIES} == counts
    stacked = np.stack([masks[c] for c in ORDINARY_CATEGORIES])
    np.testing.assert_array_equal(stacked.sum(0), 1)  # each cell exactly once
    assert masks["metadata_source_rows"].sum() == len(source.rows)
    assert masks["metadata_other_rows"].sum() == 8 - len(source.rows)
    rng = np.random.default_rng(0)
    a = rng.random((8, 4))
    out = aggregate_by_category(a, a + 1, rng.random((2, 4)), rng.random((2, 4)),
                                source, labels, obs, num_variables=3,
                                has_metadata=True, d_model=4, stage_median=1.0)  # fmt: skip
    for c, n in counts.items():
        assert out[c]["num_destinations"] == n
        assert np.isnan(out[c]["mean_abs_l2"]) == (n == 0)
    assert out["query_cells"]["num_destinations"] == 6
    assert out["query_metadata"]["num_destinations"] == 2


# 9. Injection decomposition -------------------------------------------------
@pytest.mark.parametrize("arch", LAYERWISE)
def test_flag_code_identity(models, item, arch):
    """The model's code under M' equals code(item 0) * M' (float64), and the
    flag-MLP input x is identical across batch items."""
    model = models[arch]
    prepared, _, _, M = prepare(item)
    masks = [M, remove_flags(M, [(11, 0)]),
             remove_flags(M, [(r, 2) for r in np.flatnonzero(item["ctx_regime"] == 2)])]  # fmt: skip
    seen = {}

    def spy(_mod, args, kwargs):
        seen["flag_code"] = kwargs["flag_code"]

    def spy_x(_mod, args):
        seen["x"] = args[0]

    stack = model.encoder.encoder
    mlp = model.encoder.context_intervened if model.encoder.share_intervention_encoder \
        else model.encoder.flag_code  # fmt: skip
    hs = [stack.register_forward_pre_hook(spy, with_kwargs=True),
          mlp.register_forward_pre_hook(spy_x)]  # fmt: skip
    try:
        _call(model, build_batch_inputs(prepared, masks), "cpu")
    finally:
        for h in hs:
            h.remove()
    x = seen["x"]
    assert all(torch.equal(x[b], x[0]) for b in range(len(masks)))
    code = seen["flag_code"]
    assert code.dtype == torch.float64
    Nc = M.shape[0]
    for b, m in enumerate(masks):
        assert torch.equal(code[b, :Nc], code[0, :Nc] * m[..., None].double())
        assert torch.count_nonzero(code[b, Nc:]) == 0


def test_injection_site_groups(models):
    for arch in LAYERWISE:
        sites = layerwise_injection_sites(models[arch])
        names = [n for n, _, _ in sites]
        groups = {n: g for n, _, g in sites}
        assert names[:3] == ["embedding", "inject_sample[0]", "inject_node[0]"]
        assert groups["embedding"] == groups["inject_sample[0]"] == "first"
        assert all(g == "later" for n, g in groups.items()
                   if n not in ("embedding", "inject_sample[0]"))  # fmt: skip
        stack = models[arch].encoder.encoder
        assert len(sites) == 1 + len(stack.inject_sample) + len(stack.inject_node)


@pytest.mark.parametrize("arch", LAYERWISE)
def test_injection_modes_feed_the_intended_codes(models, item, arch):
    model = models[arch]
    layout = check_supported_model(model)
    prepared, _, _, M = prepare(item)
    source = single_source(item, 11, 0)
    variants = make_variants(M, source, INJECTION_MODES)
    perturbed = remove_flags(M, source.cells)
    received: dict[str, torch.Tensor] = {}
    sites = [(n, m) for n, m, _g in layerwise_injection_sites(model)[1:]]
    Nc = M.shape[0]
    input_masks = torch.stack([M] + [v.input_mask for v in variants])
    later_masks = torch.stack([M] + [v.later_injection_mask for v in variants])
    batch = build_batch_inputs(prepared, list(input_masks))
    handles = []
    try:
        with later_injection_override(model, input_masks, later_masks):
            for name, module in sites:  # after the override hooks: sees their output
                handles.append(module.register_forward_pre_hook(
                    lambda _m, a, name=name: received.__setitem__(name, a[0].clone())))  # fmt: skip
            _call(model, batch, "cpu")
    finally:
        for h in handles:
            h.remove()
    code0 = received["inject_sample[0]"][0, :Nc]
    pert = code0 * perturbed[..., None].double()
    want = {  # (first-group code, later-group code) per injection mode
        "all_injections": (pert, pert),
        "first_injection_only": (pert, code0),
        "later_injections_only": (code0, pert),
    }
    for b, v in enumerate(variants, start=1):
        first, later = want[v.injection_mode]
        for name, _m, group in layerwise_injection_sites(model)[1:]:
            got = received[name][b, :Nc]
            assert torch.equal(got, first if group == "first" else later), (v.injection_mode, name)  # fmt: skip
    assert hook_count(model) == hook_count(models[arch])  # nothing leaked
    # decomposition runs leave the all_injections item untouched
    alone, _ = run_batch(model, prepared, variants[:1], M, device="cpu", layout=layout)
    mixed, _ = run_batch(model, prepared, variants, M, device="cpu", layout=layout)
    for s in alone:
        torch.testing.assert_close(mixed[s]["context"]["abs_l2"][1],
                                   alone[s]["context"]["abs_l2"][1], rtol=0, atol=1e-12)  # fmt: skip


@pytest.mark.parametrize("arch", LAYERWISE)
def test_decomposition_stage_structure(models, item, arch):
    """later_injections_only cannot change anything before the first later
    site; first_injection_only matches all_injections up to block 0's
    sample attention."""
    model = models[arch]
    layout = check_supported_model(model)
    prepared, _, _, M = prepare(item)
    variants = make_variants(M, single_source(item, 11, 0), INJECTION_MODES)
    states, _ = run_batch(model, prepared, variants, M, device="cpu", layout=layout)
    for s in ("embed", "block0/sample"):
        for part in ("context", "query"):
            a = states[s][part]["abs_l2"]
            assert a[3].max() <= BATCH_ROUNDING  # later_injections_only
            torch.testing.assert_close(a[2], a[1], rtol=0, atol=1e-12)
    assert torch.count_nonzero(states["block0/node"]["context"]["abs_l2"][3]) > 0


def test_override_rejects_bad_batches(models, item):
    model = models["layerwise_shared"]
    _, _, _, M = prepare(item)
    P = remove_flags(M, [(11, 0)])
    with pytest.raises(ValueError, match="baseline"):
        with later_injection_override(model, torch.stack([P, M]), torch.stack([M, M])):
            pass
    with pytest.raises(ValueError, match="only remove"):
        with later_injection_override(model, torch.stack([P, P]), torch.stack([P, M])):
            pass


# 10. Padding (aggregation level) ---------------------------------------------
def test_padding_does_not_change_aggregates():
    mask, labels = _toy_layout()
    obs = observational_rows(mask, labels)
    source = FlagSource(0, "single_flag", 0, (4,), 1)
    rng = np.random.default_rng(1)
    ca, co = rng.random((8, 4)), rng.random((8, 4)) + 0.5
    qa, qo = rng.random((2, 4)), rng.random((2, 4)) + 0.5
    kw = dict(has_metadata=True, d_model=4)
    med = stage_reference_l2(co[:, :3], np.ones((8, 3), bool))
    ref = aggregate_by_category(ca, co, qa, qo, source, labels, obs,
                                num_variables=3, stage_median=med, **kw)  # fmt: skip

    # pad: 3 rows, 1 variable (before the metadata column), 2 queries
    def pad(a, rows, cols):
        out = np.full((a.shape[0] + rows, a.shape[1] + cols), 1e6)
        out[: a.shape[0], :3] = a[:, :3]
        out[: a.shape[0], -1] = a[:, -1]
        return out

    pl = np.concatenate([labels, [7, 7, 7]])
    pobs = np.concatenate([obs, [True, True, True]])
    valid_rows = np.arange(11) < 8
    valid_vars = np.arange(4) < 3
    valid_cells = valid_rows[:, None] & valid_vars[None, :]
    pmed = stage_reference_l2(pad(co, 3, 1)[:, :4], valid_cells)
    assert pmed == med
    padded = aggregate_by_category(
        pad(ca, 3, 1), pad(co, 3, 1), pad(qa, 2, 1), pad(qo, 2, 1), source, pl,
        pobs, num_variables=4, stage_median=pmed, valid_rows=valid_rows,
        valid_variables=valid_vars, valid_queries=np.arange(4) < 2, **kw,
    )  # fmt: skip
    assert padded.keys() == ref.keys()
    for c in ref:
        for k, v in ref[c].items():
            np.testing.assert_equal(padded[c][k], v, err_msg=f"{c}/{k}")


# 11. Hook lifecycle -----------------------------------------------------------
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_hooks_removed_even_on_error(models, item, arch):
    model = models[arch]
    before = hook_count(model)
    sources = select_sources(item, sources_per_context=1, regime_sources_per_context=1,
                             ablation="both", seed=0, context_id=0)  # fmt: skip
    analyse_context(model, item, sources, device="cpu")
    assert hook_count(model) == before

    def boom(*_):
        raise RuntimeError("boom")

    h = model.encoder.encoder.node_layers[1].register_forward_hook(boom)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            analyse_context(model, item, sources, device="cpu")
    finally:
        h.remove()
    assert hook_count(model) == before


# 12. Output-level metrics -----------------------------------------------------
def test_monte_carlo_kl_matches_closed_form():
    g = torch.Generator().manual_seed(0)
    n_t = 4
    mu0 = torch.randn(n_t, 1, generator=g, dtype=torch.float64)
    sd0 = 0.5 + torch.rand(n_t, 1, generator=g, dtype=torch.float64)
    mu1 = mu0[None] + 0.3 * torch.randn(2, n_t, 1, generator=g, dtype=torch.float64)
    sd1 = sd0[None] * (1 + 0.3 * torch.rand(2, n_t, 1, generator=g, dtype=torch.float64))
    lw = torch.zeros(n_t, 1, dtype=torch.float64)
    exact = compute_predictive_changes((mu0, sd0, lw), (mu1, sd1, lw[None].repeat(2, 1, 1)))
    assert exact["kl_method"] == "closed_form"
    assert (exact["predictive_kl"] > 0).all() and (exact["kl_standard_error"] == 0).all()
    # the same Gaussians written as two identical components -> Monte Carlo
    two = lambda a: torch.cat([a, a], -1)  # noqa: E731
    lw2 = torch.full((n_t, 2), np.log(0.5), dtype=torch.float64)
    mc = compute_predictive_changes(
        (two(mu0), two(sd0), lw2), (two(mu1), two(sd1), lw2[None].repeat(2, 1, 1)),
        num_samples=20000, seed=3,
    )  # fmt: skip
    assert mc["kl_method"] == "monte_carlo"
    err = np.abs(mc["predictive_kl"] - exact["predictive_kl"])
    assert (err < 5 * mc["kl_standard_error"] + 1e-3).all()
    np.testing.assert_allclose(mc["mean_shift"], exact["mean_shift"], atol=1e-12)
    np.testing.assert_allclose(mc["std_shift"], exact["std_shift"], atol=1e-12)


# 15. Undefined metrics ---------------------------------------------------------
def test_undefined_metrics_are_nan():
    empty = summarise_tokens(np.ones(3), np.ones(3), np.zeros(3, bool),
                             stage_median=1.0, d_model=4)  # fmt: skip
    assert empty["num_destinations"] == 0
    assert all(np.isnan(v) for k, v in empty.items() if k != "num_destinations")
    zero_base = summarise_tokens(np.ones(2), np.zeros(2), np.ones(2, bool),
                                 stage_median=0.0, d_model=4)  # fmt: skip
    assert np.isnan(zero_base["mean_rel_l2"])
    assert np.isnan(zero_base["mean_stage_normalised_l2"])
    assert zero_base["mean_abs_l2"] == 1.0
    assert np.isnan(source_normalised(1.0, 0.0))
    assert np.isnan(source_normalised(1.0, 1e-9))
    assert source_normalised(1.0, 2.0) == pytest.approx(0.5)
    # D = 1: no "other variables" destinations
    labels = np.array([0, 1, 1])
    mask = np.array([[False], [True], [True]])
    masks = context_category_masks(FlagSource(0, "regime", 0, (1, 2), 1), labels,
                                   observational_rows(mask, labels), num_variables=1)  # fmt: skip
    assert masks["source_rows_other_variables"].sum() == 0
    assert masks["other_rows_other_variables"].sum() == 0


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_run_has_no_infinities(models, arch):
    res = run_flag_propagation(models[arch], bank_items(2), device="cpu")
    s, p = res["summary"], res["predictions"]
    num = s.select_dtypes("number")
    assert not np.isinf(num.to_numpy()).any()
    assert not np.isinf(p.select_dtypes("number").to_numpy()).any()
    later = s[s.injection_mode == "later_injections_only"]
    if len(later):  # zero initial magnitude -> NaN, not inf or 0
        assert later.source_normalised_sensitivity.isna().all()
    regime = s[s.ablation_mode == "regime"]
    assert regime.row_becomes_observational.all()
    assert regime[regime.category == "regime_mates_same_variable"].mean_abs_l2.isna().all()


# Architecture checks ----------------------------------------------------------
def test_unsupported_architecture_and_variable_count(tmp_path, models, item):
    r1 = build_model(tmp_path, "r1")
    with pytest.raises(ValueError, match="'r1'"):
        run_flag_propagation(r1, [item], device="cpu")
    big = dict(item, ctx_x=np.zeros((14, D + 1), np.float32),
               ctx_int=np.eye(14, D + 1, dtype=bool))  # fmt: skip
    big.pop("ctx_regime")
    with pytest.raises(ValueError, match="num_nodes"):
        analyse_context(models["extra_node"], big, [], device="cpu")


# Noise floor (identical-mask control item) ----------------------------------
def _floor_run(model, item):
    sources = select_sources(item, sources_per_context=2, regime_sources_per_context=1,
                             ablation="both", seed=0, context_id=0)  # fmt: skip
    res = analyse_context(model, item, sources, device="cpu", max_batch_size=4)
    return pd.DataFrame(res["summary"]), pd.DataFrame(res["predictions"])


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_noise_floor_float64(models, item, arch):
    s, p = _floor_run(models[arch], item)
    for col in ("noise_floor_mean_abs_l2", "noise_floor_mean_stage_normalised_l2"):
        assert s[col].notna().any()
        assert s[col].max() <= BATCH_ROUNDING, col
    # empty categories stay NaN in the floor as in the signal
    assert (s.mean_abs_l2.isna() == s.noise_floor_mean_abs_l2.isna()).all()
    assert p.noise_floor_predictive_kl.abs().max() <= 1e-9


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_noise_floor_float32(tmp_path, item, arch):
    model = build_model(tmp_path, arch, dtype=torch.float32)
    s, p = _floor_run(model, item)
    floor = s.noise_floor_mean_stage_normalised_l2  # relative to baseline norms
    assert np.isfinite(floor.dropna()).all() and floor.max() < 1e-4
    assert np.isfinite(p.noise_floor_predictive_kl).all()
    assert p.noise_floor_predictive_kl.abs().max() < 1e-2


def test_control_item_is_last_and_counts_towards_batch_size(models, item):
    model = models["layerwise_shared"]
    seen = []

    def spy(_mod, _args, kwargs):
        seen.append(kwargs["context_int_mask"].clone())

    sources = select_sources(item, sources_per_context=3, regime_sources_per_context=1,
                             ablation="both", seed=0, context_id=0)  # fmt: skip
    h = model.register_forward_pre_hook(spy, with_kwargs=True)
    try:
        analyse_context(model, item, sources, device="cpu", max_batch_size=5)
    finally:
        h.remove()
    M = torch.as_tensor(np.asarray(item["ctx_int"]))
    num_variants = len(sources) * 3  # three injection modes per source
    assert len(seen) == -(-num_variants // 3)  # max_batch_size 5 = 3 variants + 2
    for masks in seen:
        assert len(masks) <= 5
        assert torch.equal(masks[0], M) and torch.equal(masks[-1], M)
    with pytest.raises(ValueError, match="max_batch_size"):
        analyse_context(model, item, sources, device="cpu", max_batch_size=2)


# Precision guard -------------------------------------------------------------
def test_autocast_is_disabled_and_half_precision_rejected(tmp_path, models, item):
    model = models["layerwise_shared"]
    states = []

    def spy(_mod, _args, out):
        states.append((torch.is_autocast_enabled("cpu"), out.dtype))

    h = model.encoder.encoder.node_layers[0].register_forward_hook(spy)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            analyse_context(model, item, [single_source(item, 11, 0)], device="cpu")
    finally:
        h.remove()
    assert states and all(not on and dt == torch.float64 for on, dt in states)
    bf16 = build_model(tmp_path, "extra_node", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float32 or float64"):
        run_flag_propagation(bf16, bank_items(1), device="cpu")


# 14. Manifest reproducibility -------------------------------------------------
@pytest.mark.parametrize("arch", ["layerwise_shared", "extra_node"])
def test_manifest_reproduces_run(models, arch):
    bank = bank_items(3)
    first = run_flag_propagation(models[arch], bank, device="cpu", seed=4)
    manifest = first["sources"]
    assert set(manifest.columns) >= {"context_id", "context_fingerprint", "regime_sizes",
                                     "num_intervened_cells", "source_rows"}  # fmt: skip
    plan = sources_from_manifest(manifest, bank)
    expected = {
        cid: select_sources(bank[cid], sources_per_context=3, regime_sources_per_context=1,
                            ablation="both", seed=4, context_id=cid)
        for cid in plan
    }  # fmt: skip
    assert plan == expected
    # selection settings are ignored when a manifest is given; the seed still
    # sets the Monte Carlo KL draws, so it is kept for paired runs
    again = run_flag_propagation(models[arch], bank, device="cpu", seed=4,
                                 max_contexts=1, sources_per_context=1,
                                 source_manifest=manifest)  # fmt: skip
    assert again["sources"].equals(manifest)
    assert again["summary"].equals(first["summary"])
    assert again["predictions"].equals(first["predictions"])


def test_invalid_manifest_fails_clearly():
    bank = bank_items(2)
    plan = {0: select_sources(bank[0], sources_per_context=2, regime_sources_per_context=1,
                              ablation="both", seed=0, context_id=0)}  # fmt: skip
    good = build_source_manifest(bank, plan)
    assert sources_from_manifest(good, bank) == plan
    mask = np.asarray(bank[0]["ctx_int"])
    unflagged = int(np.flatnonzero(~mask.any(1))[0])

    def broken(**change):
        m = good.copy()
        for k, v in change.items():
            m.loc[0 if k != "regime_rows" else len(m) - 1,
                  "source_rows" if k == "regime_rows" else k] = v
        return m

    cases = [
        (broken(context_fingerprint="0" * 16), "context differs"),
        (broken(context_id=5), "outside the bank"),
        (broken(query_outcome=(int(bank[0]["Y"]) + 1) % D), "query"),
        (broken(source_rows=str(unflagged)), "no flag"),
        (broken(ablation_mode="cells"), "unknown ablation_mode"),
        (broken(source_regime=99), "regime 99"),
        (broken(regime_rows=good.source_rows.iloc[-1].split(";")[0]), "all flagged rows"),
    ]
    for manifest, message in cases:
        with pytest.raises(ValueError, match=message):
            sources_from_manifest(manifest, bank)


def test_net_change_columns(models):
    s = run_flag_propagation(models["extra_node"], bank_items(1), device="cpu")["summary"]
    g = s[(s.source_id == 0) & (s.category == "metadata_source_rows")].sort_values("stage_index")
    assert np.isnan(g.net_change_mean_abs_l2.iloc[0])
    np.testing.assert_allclose(g.net_change_mean_abs_l2.iloc[1:], np.diff(g.mean_abs_l2))


# Smoke test: the CLI end to end ----------------------------------------------
OUTPUT_FILES = ["flag_propagation_summary.csv", "flag_propagation_predictions.csv",
                "flag_propagation_sources.csv", "flag_propagation_metadata.json",
                "flag_propagation_examples.npz", "flag_propagation_A_example0.png",
                "flag_propagation_B_curves.png", "flag_propagation_E_predictive_kl.png",
                "flag_propagation_F_net_change.png"]  # fmt: skip


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cli_smoke(tmp_path, precision):
    import json

    from flag_propagation_spec import write_bank, write_checkpoint

    from CITNP.evaluation.__main__ import main

    bank = write_bank(tmp_path / "bank.pkl", n_contexts=3)
    manifest = tmp_path / "manifest.csv"
    runs = []
    for i, arch in enumerate(["layerwise_shared", "layerwise_separate", "extra_node"]):
        ckpt, cfg = write_checkpoint(tmp_path, arch)
        out = tmp_path / arch
        args = ["flag-propagation", "--ckpt", ckpt, "--model_config", cfg,
                "--bank", bank, "--device", "cpu", "--max_contexts", "3",
                "--save_examples", "--num_examples", "2", "--max_batch_size", "6",
                "--kl_samples", "200", "--out", str(out)]  # fmt: skip
        args += ["--float64"] if precision == "float64" else []
        args += (["--save_source_manifest", str(manifest)] if i == 0
                 else ["--source_manifest", str(manifest)])  # fmt: skip
        main(args)
        for name in OUTPUT_FILES:
            assert (out / name).exists(), (arch, name)
        layerwise = arch.startswith("layerwise")
        assert (out / "flag_propagation_D_injection.png").exists() == layerwise
        meta = json.loads((out / "flag_propagation_metadata.json").read_text())
        assert meta["dtype"] == precision and meta["autocast"] == "disabled"
        assert meta["architecture"] == ("layerwise" if layerwise else "extra_node")
        assert meta["share_intervention_encoder"] == (
            {"layerwise_shared": True, "layerwise_separate": False}.get(arch)
        )
        s = pd.read_csv(out / "flag_propagation_summary.csv")
        for col in ("noise_floor_mean_abs_l2", "noise_floor_mean_stage_normalised_l2",
                    "input_mask", "later_injection_mask", "net_change_mean_abs_l2"):
            assert col in s.columns
        runs.append(str(out))
    assert manifest.exists()
    main(["flag-propagation", "--plot_only", *runs, "--out", str(tmp_path / "compare")])
    for name in ("flag_propagation_C_architectures.png", "flag_propagation_E_predictive_kl.png"):
        assert (tmp_path / "compare" / name).exists()
    main(["flag-propagation", "--plot_only", runs[0]])  # single-run replot
