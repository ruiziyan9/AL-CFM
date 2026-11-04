"""Tests for the label-sensitivity context edits (intervention_diagnostics.
run_context_edits, summarise_context_edits and the mask edits), the
observational-identifiability label, metrics.sample_mixture,
predict_mixture's normalisation_int_mask and the `context-edits` command
(random tiny models or stub predictors, CPU, tasks from a small prior). No
thresholds on what a model has learned."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.evaluation.intervention_diagnostics import (
    ALL,
    CONTEXT_EDIT_STRATA,
    CONTEXT_EDITS,
    draw_new_targets,
    edit_counts,
    edit_order,
    move_row_targets,
    relabel_rows_as_observational,
    run_context_edits,
    summarise_context_edits,
)
from CITNP.evaluation.metrics import sample_mixture
from CITNP.evaluation.model_api import predict_mixture
from CITNP.evaluation.oracle import LinearSCM, all_dags, dag_key
from CITNP.evaluation.tasks import observationally_identified

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_intervention_probes import (  # noqa: E402
    CFG,
    EDIT_BASE_SEED,
    PRIOR,
    tasks_with_interventions,
    tiny_model,
    tiny_model_yaml,
    window_with_interventions,
)
from test_mixed_context_model import D  # noqa: E402

NUM_TASKS = 6


def small_run() -> dict:
    """Six tasks that have intervened rows (selected, not left to chance)."""
    return dict(
        D=D, device="cpu", n_mc=20, n_y=8, max_query_points=4,
        task_indices=list(tasks_with_interventions(EDIT_BASE_SEED, NUM_TASKS)),
    )  # fmt: skip


def uniform_prior(num_variables: int) -> dict:
    dags = all_dags(num_variables)
    return {dag_key(g): -np.log(len(dags)) for g in dags}


def example_mask(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((30, D), dtype=bool)
    mask[:6, 0] = True
    mask[6:10, 2] = True
    return mask[rng.permutation(30)]


# --------------------------------------------------------------------------- #
# Mask edits
# --------------------------------------------------------------------------- #
def test_edit_order_and_counts():
    mask = example_mask()
    order = edit_order(mask, np.random.default_rng(0))
    assert sorted(order) == sorted(np.flatnonzero(mask.any(1)))
    assert np.array_equal(order, edit_order(mask, np.random.default_rng(0)))
    assert edit_counts(1) == [1]
    assert edit_counts(4) == [1, 2, 4]
    assert edit_counts(5) == [1, 2, 4, 5]
    assert edit_counts(100) == [1, 2, 4, 8, 16, 32, 64, 100]
    assert edit_counts(0) == []


def test_relabel_rows_as_observational():
    mask = example_mask()
    order = edit_order(mask, np.random.default_rng(0))
    for n in edit_counts(len(order)):
        edited = relabel_rows_as_observational(mask, order[:n])
        assert edited.shape == mask.shape and edited.dtype == bool
        assert not edited[order[:n]].any()
        np.testing.assert_array_equal(edited[order[n:]], mask[order[n:]])
        assert edited.any(1).sum() == len(order) - n
    assert mask.any(1).sum() == len(order)  # input untouched


def test_move_row_targets():
    mask = example_mask()
    moved_from, moved_to = draw_new_targets(mask, np.random.default_rng(1))
    intervened = mask.any(1)
    assert (moved_from[~intervened] == -1).all()
    assert (moved_from[intervened] >= 0).all()
    assert (moved_from[intervened] != moved_to[intervened]).all()
    assert mask[intervened, moved_from[intervened]].all()
    order = edit_order(mask, np.random.default_rng(0))
    for n in (1, len(order)):
        edited = move_row_targets(mask, order[:n], moved_from, moved_to)
        assert edited.shape == mask.shape and edited.dtype == bool
        np.testing.assert_array_equal(edited.sum(1), mask.sum(1))
        for r in order[:n]:
            assert edited[r, moved_to[r]] and not edited[r, moved_from[r]]
        np.testing.assert_array_equal(edited[order[n:]], mask[order[n:]])


def test_new_target_roles():
    from CITNP.evaluation.intervention_diagnostics import new_target_roles

    moved_to = np.array([0, 2, -1, 1, 2])
    assert new_target_roles(moved_to, [1, 4], T=2, Y=0) == {
        "new_target_role": "treatment", "num_moved_to_treatment": 2,
        "num_moved_to_outcome": 0, "num_moved_to_other": 0,
    }  # fmt: skip
    assert new_target_roles(moved_to, [0], T=2, Y=0)["new_target_role"] == (
        "outcome"
    )
    assert new_target_roles(moved_to, [3], T=2, Y=0)["new_target_role"] == "other"
    mixed = new_target_roles(moved_to, [0, 1, 3], T=2, Y=0)
    assert mixed["new_target_role"] == "mixed"
    assert new_target_roles(moved_to, [2], T=2, Y=0)["new_target_role"] == (
        "not_applicable"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_observational_identifiability():
    chain = np.zeros((3, 3))
    chain[0, 1], chain[1, 2] = 0.9, -0.8
    chain_scm = LinearSCM(chain, np.array([0.3, 0.2, 0.25]))
    # MEC {0->1->2, 2->1->0, 0<-1->2}: the effect of X0 on X2 is not identified
    assert not observationally_identified(chain_scm, 0, 2)
    collider = np.zeros((3, 3))
    collider[0, 2], collider[1, 2] = 0.9, 0.7
    collider_scm = LinearSCM(collider, np.array([0.3, 0.3, 0.2]))
    # a v-structure is alone in its MEC: every query is identified
    assert observationally_identified(collider_scm, 0, 2)
    assert observationally_identified(collider_scm, 2, 0)


def test_sample_mixture_moments():
    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 4.0], [1.0, 1.0]])
    sd = np.array([[1.0, 1.0], [0.5, 0.5]])
    logw = np.log(np.array([[0.25, 0.75], [0.5, 0.5]]))
    y = sample_mixture(mu, sd, logw, 20000, rng)
    assert y.shape == (2, 20000)
    np.testing.assert_allclose(y.mean(1), [3.0, 1.0], atol=0.05)


def test_kl_standard_error():
    from CITNP.evaluation.metrics import kl_mc, kl_mc_mean_with_standard_error

    rng = np.random.default_rng(0)
    lp_p = torch.as_tensor(rng.normal(size=(4, 50)))
    lp_q = torch.as_tensor(rng.normal(size=(4, 50)))
    kl, se = kl_mc_mean_with_standard_error(lp_p, lp_q)
    assert kl == pytest.approx(float(kl_mc(lp_p, lp_q).mean()))
    d = (lp_p - lp_q).numpy()
    expected = np.sqrt((d.var(1, ddof=1) / 50).sum()) / 4
    assert se == pytest.approx(expected)
    # identical densities: zero KL, zero standard error
    assert kl_mc_mean_with_standard_error(lp_p, lp_p) == (0.0, 0.0)


def test_normalisation_mask_default_and_override():
    model = tiny_model("r1")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, D))
    mask = example_mask()
    grid = np.linspace(-1, 1, 3)
    default = predict_mixture(model, x, mask, 0, 2, grid, device="cpu")
    explicit = predict_mixture(
        model, x, mask, 0, 2, grid, device="cpu", normalisation_int_mask=mask
    )
    other = predict_mixture(
        model, x, mask, 0, 2, grid, device="cpu",
        normalisation_int_mask=np.zeros_like(mask),
    )  # fmt: skip
    for a, b in zip(default, explicit):
        assert torch.equal(a, b)
    assert not torch.equal(default[0], other[0])


def test_normalisation_mask_freezes_every_scale_dependent_input():
    """With the unedited mask as normalisation_int_mask, an edited mask
    changes context_int_mask only: the standardised values, do-values,
    scale features and the raw-scale map stay those of the unedited input."""
    from CITNP.evaluation.model_api import _model_inputs

    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, D)) * 3 + 1
    mask = example_mask()
    assert mask.any()
    edited = np.zeros_like(mask)
    kw = dict(zero_flags=False, normalise=True)
    t = np.linspace(-1, 1, 3)
    before, mean_b, std_b = _model_inputs(x, mask, [0, 1], [2, 2], t, **kw)
    frozen, mean_f, std_f = _model_inputs(
        x, edited, [0, 1], [2, 2], t, normalisation_int_mask=mask, **kw
    )
    free, mean_e, std_e = _model_inputs(x, edited, [0, 1], [2, 2], t, **kw)
    assert torch.equal(mean_b, mean_f) and torch.equal(std_b, std_f)
    for key in ("context_data", "target_data", "context_scale"):
        assert torch.equal(before[key], frozen[key]), key
    assert not torch.equal(before["context_int_mask"], frozen["context_int_mask"])
    assert not torch.equal(before["context_scale"], free["context_scale"])
    assert not torch.equal(mean_b, mean_e)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
class RecordingPredictor:
    """Stub of `predict_mixture_versions`: N(c * number of intervened cells
    of the version, 1) for every query (c = 0 makes it ignore the labels);
    records its inputs."""

    def __init__(self, mask_weight: float):
        self.mask_weight = mask_weight
        self.calls: list = []

    def __call__(self, x, masks, Ts, Ys, t, normalisation_int_masks=None):
        self.calls.append((x.copy(), masks.copy(), normalisation_int_masks))
        V, Q, n = len(masks), len(Ts), t.shape[1]
        assert t.shape[0] == Q
        means = self.mask_weight * masks.reshape(V, -1).sum(1).astype(float)
        mu = torch.as_tensor(means)[:, None, None, None].expand(V, Q, n, 1)
        sd = torch.ones(V, Q, n, 1, dtype=torch.float64)
        return mu.clone(), sd, torch.zeros(V, Q, n, 1, dtype=torch.float64)


def test_one_batched_call_per_context_with_valid_masks():
    predictor = RecordingPredictor(mask_weight=0.1)
    df = run_context_edits(
        predictor, CFG, log_graph_prior=uniform_prior(D), **small_run()
    )
    # every selected task is scored, with one model call holding every
    # version of its context
    assert df["task"].nunique() == NUM_TASKS
    assert len(predictor.calls) == NUM_TASKS
    for x, masks, normalisation in predictor.calls:
        assert normalisation is None  # the model's normal input path
        original = masks[0]
        num_intervened = int(original.any(1).sum())
        assert len(masks) == 1 + len(CONTEXT_EDITS) * len(
            edit_counts(num_intervened)
        )
        assert masks.dtype == bool and masks.shape[1:] == original.shape
        for m in masks:
            # each row keeps its number of intervened cells or loses all
            counts = m.sum(1)
            assert ((counts == original.sum(1)) | (counts == 0)).all()
    edited_all = df[df["fraction_edited_rows"] == 1]
    assert set(edited_all["edit"]) == set(CONTEXT_EDITS)


def test_keep_original_normalisation_ablation():
    predictor = RecordingPredictor(mask_weight=0.1)
    run_context_edits(
        predictor, CFG, log_graph_prior=uniform_prior(D),
        keep_original_normalisation=True, **small_run(),
    )  # fmt: skip
    for _, masks, normalisation in predictor.calls:
        assert normalisation.shape == masks.shape
        for version in normalisation:
            np.testing.assert_array_equal(version, masks[0])


def test_batched_versions_match_single_predictions():
    from CITNP.evaluation.model_api import predict_mixture_versions

    model = tiny_model("extra_node")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, D))
    mask = example_mask()
    masks = np.stack([mask, np.zeros_like(mask), mask[:, [1, 2, 0]]])
    Ts, Ys = np.array([0, 0, 2]), np.array([1, 2, 1])
    t = rng.normal(size=(3, 4))
    for normalisation in (None, np.repeat(mask[None], 3, axis=0)):
        batched = predict_mixture_versions(
            model, x, masks, Ts, Ys, t, device="cpu",
            normalisation_int_masks=normalisation, max_batch_size=4,
        )  # fmt: skip
        for v in range(3):
            for q in range(3):
                single = predict_mixture(
                    model, x, masks[v], Ts[q], Ys[q], t[q], device="cpu",
                    normalisation_int_mask=None
                    if normalisation is None
                    else normalisation[v],
                )  # fmt: skip
                for b, s_ in zip(batched, single):
                    np.testing.assert_allclose(
                        b[v, q].numpy(), s_[0].numpy(), rtol=1e-5, atol=1e-6
                    )


def test_label_blind_model_has_zero_sensitivity():
    df = run_context_edits(
        RecordingPredictor(mask_weight=0.0), CFG,
        log_graph_prior=uniform_prior(D), **small_run(),
    )  # fmt: skip
    assert (df["model_mean_shift_rms"] == 0).all()
    assert (df["model_kl_original_to_edited"] == 0).all()
    summary = summarise_context_edits(df)
    pooled = summary[
        (summary["identifiable_from_observational"] == ALL)
        & (summary["treatment_intervened_in_context"] == ALL)
        & (summary["new_target_role"] == ALL)
        & (summary["num_edited_rows"] == ALL)
    ]
    finite = pooled["mean_shift_ratio"].dropna()
    assert len(finite) and (finite == 0).all()


@pytest.mark.parametrize("encoding", ["extra_node", "layerwise"])
def test_run_context_edits_structure(encoding):
    df = run_context_edits(
        tiny_model(encoding), CFG, log_graph_prior=uniform_prior(D),
        **small_run(),
    )  # fmt: skip
    assert df["task"].nunique() == NUM_TASKS
    assert set(df["edit"]) == set(CONTEXT_EDITS)
    assert set(df.columns) >= {
        "task", "T", "Y", "edit", "num_edited_rows", "num_intervened_rows",
        "fraction_edited_rows", "num_edited_rows_targeting_treatment",
        "relation", "identifiable_from_observational",
        "treatment_intervened_in_context", "outcome_intervened_in_context",
        "model_mean_shift", "oracle_mean_shift",
        "model_mean_shift_rms", "oracle_mean_shift_rms",
        "model_slope_change", "oracle_slope_change",
        "model_kl_original_to_edited", "oracle_kl_original_to_edited",
        "mean_shift_sign_agreement",
        "secondary_kl_oracle_to_model_original",
        "secondary_kl_oracle_to_model_edited",
        "model_kl_original_to_edited_se", "oracle_kl_original_to_edited_se",
        "secondary_kl_oracle_to_model_original_se",
        "secondary_kl_oracle_to_model_edited_se",
    }  # fmt: skip
    for column in [c for c in df.columns if c.endswith("_se")]:
        assert (df[column] >= 0).all(), column
    assert not any("loglik" in c for c in df.columns)
    relabel = df[df["edit"] == "relabel_observational"]
    assert (relabel["new_target_role"] == "not_applicable").all()
    moved = df[df["edit"] == "move_target"]
    moved_counts = moved[
        ["num_moved_to_treatment", "num_moved_to_outcome", "num_moved_to_other"]
    ].sum(1)
    assert (moved_counts == moved["num_edited_rows"]).all()
    single = moved[moved["num_edited_rows"] == 1]
    assert set(single["new_target_role"]) <= {"treatment", "outcome", "other"}
    assert (df["T"] != df["Y"]).all()
    assert (df["num_edited_rows"] >= 1).all()
    assert (df["num_edited_rows"] <= df["num_intervened_rows"]).all()
    for column in ("model_mean_shift_rms", "oracle_mean_shift_rms",
                   "model_kl_original_to_edited",
                   "oracle_kl_original_to_edited"):  # fmt: skip
        assert np.isfinite(df[column]).all(), column

    summary = summarise_context_edits(df)
    assert list(summary.columns[: len(CONTEXT_EDIT_STRATA) + 2]) == [
        "edit", *CONTEXT_EDIT_STRATA, "num_edited_rows"
    ]
    assert set(summary["edit"]) == set(CONTEXT_EDITS)
    pooled = summary[
        (summary["identifiable_from_observational"] == ALL)
        & (summary["treatment_intervened_in_context"] == ALL)
        & (summary["new_target_role"] == ALL)
    ]
    overall = pooled[pooled["num_edited_rows"] == ALL]
    assert len(overall) == len(CONTEXT_EDITS)
    assert (overall["num_queries"] == len(df) // len(CONTEXT_EDITS)).all()
    curve = pooled[pooled["num_edited_rows"] != ALL]
    assert set(curve["num_edited_rows"]) == set(df["num_edited_rows"])


def test_context_edits_cli(tmp_path):
    from CITNP.evaluation.__main__ import main

    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": tiny_model("r1").state_dict()}, ckpt)
    model_config = tiny_model_yaml(tmp_path / "model.yaml", "r1")
    data_config = tmp_path / "prior.yaml"
    data_config.write_text(yaml.safe_dump(PRIOR))
    out = tmp_path / "edits.csv"
    main(["context-edits", "--ckpt", str(ckpt), "--model_config",
          str(model_config), "--data_config", str(data_config),
          "--n_tasks", "2",
          "--task_offset", str(window_with_interventions(EDIT_BASE_SEED, 2)),
          "--n_mc", "20", "--n_y", "8",
          "--max_query_points", "4", "--max_batch_size", "8",
          "--graph_prior_samples", "200", "--device", "cpu",
          "--out", str(out)])  # fmt: skip
    assert pd.read_csv(out)["task"].nunique() == 2
    assert len(pd.read_csv(tmp_path / "edits_summary.csv")) > 0
    meta = json.loads((tmp_path / "edits_meta.json").read_text())
    assert meta["num_variables"] == D
