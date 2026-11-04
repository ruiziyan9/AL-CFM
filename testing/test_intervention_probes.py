"""Tests for the per-layer intervention probes: probes.fit_classification_probe
and friends, intervention_diagnostics.run_intervention_probes on every
intervention encoding, and the `probe-intervention` command (random tiny
models, CPU, tasks from a small prior). No thresholds on what a model has
learned."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.mixed_regime_generator import MixedRegimeConfig
from CITNP.evaluation.intervention_diagnostics import (
    CELL_PROBES,
    INTERVENTION_PROBES,
    METADATA_PROBES,
    VALUE_BASELINE_POINT,
    choose_question,
    pick_rows,
    run_intervention_probes,
)
from CITNP.evaluation.model_api import probe_points
from CITNP.evaluation.probes import (
    assign_group_folds,
    fit_classification_probe,
    multiclass_balanced_accuracy,
    shuffle_labels,
)
from CITNP.evaluation.tasks import load_prior_config, prior_task
from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    resolve_model_class,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import D, tiny_config  # noqa: E402

# Small linear-Gaussian prior (also used by the context-edit tests).
PRIOR = dict(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    k_max=2,
    n_ctx_range=[30, 40],
    n_qry=6,
    qry_mode="grouped",
    n_obs_min=10,
)
CFG = MixedRegimeConfig(**PRIOR)
ENCODINGS = sorted(INTERVENTION_ENCODINGS)
RESULT_COLUMNS = {
    "probe", "point", "depth", "labels", "accuracy", "balanced_accuracy",
    "auc", "majority_class_rate", "n", "num_classes", "num_groups",
}  # fmt: skip
SMALL_RUN = dict(D=D, device="cpu", n_tasks=6, rows_per_task=8, n_folds=2)


def tiny_model(encoding: str, seed: int = 0) -> torch.nn.Module:
    model_class, _ = resolve_model_class(encoding)
    torch.manual_seed(seed)
    return model_class(tiny_config()).eval()


def tiny_model_yaml(path: Path, encoding: str) -> Path:
    cfg = tiny_config()
    path.write_text(
        yaml.safe_dump({
            "d_model": cfg.d_model, "nhead": cfg.nhead,
            "num_layers_encoder": cfg.num_layers_encoder,
            "sample_attn_mode": cfg.sample_attn_mode,
            "dim_feedforward": cfg.dim_feedforward,
            "emb_depth": cfg.emb_depth,
            "num_mixture_components": cfg.num_mixture_components,
            "intervention_encoding": encoding,
        })  # fmt: skip
    )
    return path


# --------------------------------------------------------------------------- #
# Probe fitting
# --------------------------------------------------------------------------- #
def test_group_folds_never_split_a_group():
    groups = np.repeat(np.arange(7), 5)
    fold = assign_group_folds(groups, 3, seed=0)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1
    assert set(fold) == {0, 1, 2}
    with pytest.raises(ValueError, match="groups"):
        assign_group_folds(groups, 8, seed=0)


def test_multiclass_probe_on_separable_and_random_labels():
    """Checks the fitting routine on synthetic data, not a model."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 20)
    X = rng.normal(size=(600, 4))
    labels = np.argmax(X[:, :3], axis=1)
    result = fit_classification_probe(X, labels, groups)
    assert result["num_classes"] == 3 and result["n"] == 600
    assert result["balanced_accuracy"] > 0.8
    assert np.isnan(result["auc"])
    control = fit_classification_probe(X, shuffle_labels(labels, 1), groups)
    assert abs(control["balanced_accuracy"] - 1 / 3) < 0.1


def test_binary_probe_reports_auc():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 10)
    X = rng.normal(size=(200, 3))
    labels = (X[:, 0] > 0).astype(int)
    result = fit_classification_probe(X, labels, groups, n_folds=4)
    assert result["auc"] > 0.9
    assert result["majority_class_rate"] >= 0.5


def test_balanced_accuracy_and_shuffle():
    labels = np.array([0, 0, 0, 1])
    assert multiclass_balanced_accuracy(np.array([0, 0, 0, 0]), labels) == 0.5
    shuffled = shuffle_labels(labels, 3)
    assert sorted(shuffled) == sorted(labels)
    assert np.array_equal(shuffled, shuffle_labels(labels, 3))


# --------------------------------------------------------------------------- #
# Task pieces
# --------------------------------------------------------------------------- #
def test_pick_rows_balances_intervened_rows():
    mask = np.zeros((30, D), dtype=bool)
    mask[:5, 1] = True
    rows = pick_rows(mask, 8, np.random.default_rng(0))
    assert len(rows) == 8 and len(set(rows)) == 8
    assert mask[rows].any(1).sum() == 4
    rows = pick_rows(mask, 20, np.random.default_rng(0))
    assert mask[rows].any(1).sum() == 5  # all intervened rows available


def test_question_is_a_query_treatment():
    task = prior_task(CFG, 0, 9201, D)
    T, Y = choose_question(task, np.random.default_rng(0))
    assert T != Y
    assert task.qry_int[:, T].any()


def test_load_prior_config_accepts_a_dataset_config(tmp_path):
    prior = tmp_path / "prior.yaml"
    prior.write_text(yaml.safe_dump(PRIOR))
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump({
            "dataset_name": "tiny", "prior_config": str(prior),
            "tasks_per_chunk_file": 4, "tasks_per_shape_group": 2,
            "splits": {"test": {"num_tasks": 4, "base_seed": 0}},
        })  # fmt: skip
    )
    assert asdict(load_prior_config(prior)) == asdict(CFG)
    assert asdict(load_prior_config(dataset)) == asdict(CFG)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_run_intervention_probes_structure(encoding):
    model = tiny_model(encoding)
    df = run_intervention_probes(model, CFG, **SMALL_RUN)
    assert set(df.columns) == RESULT_COLUMNS
    expected = set(CELL_PROBES)
    if encoding == "extra_node":
        expected |= set(METADATA_PROBES)
    assert set(df["probe"]) == expected <= set(INTERVENTION_PROBES)
    assert set(df["labels"]) == {"true", "shuffled"}
    names = probe_points(model)
    for probe, rows in df.groupby("probe"):
        points = list(names)
        if probe in CELL_PROBES:
            points.append(VALUE_BASELINE_POINT)
        assert sorted(rows["point"]) == sorted(2 * points), probe
        depth = rows.set_index(["point", "labels"])["depth"]
        for d, name in enumerate(names):
            assert depth[(name, "true")] == d
    scores = df["balanced_accuracy"].dropna()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_run_intervention_probes_is_deterministic():
    model = tiny_model("extra_node")
    first = run_intervention_probes(model, CFG, **SMALL_RUN)
    second = run_intervention_probes(model, CFG, **SMALL_RUN)
    pd.testing.assert_frame_equal(first, second)


def test_probe_intervention_cli(tmp_path):
    from CITNP.evaluation.__main__ import main

    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": tiny_model("extra_node").state_dict()}, ckpt)
    model_config = tiny_model_yaml(tmp_path / "model.yaml", "extra_node")
    data_config = tmp_path / "prior.yaml"
    data_config.write_text(yaml.safe_dump(PRIOR))
    out = tmp_path / "probes.csv"
    main(["probe-intervention", "--ckpt", str(ckpt), "--model_config",
          str(model_config), "--data_config", str(data_config),
          "--n_tasks", "4", "--rows_per_task", "8", "--n_folds", "2",
          "--device", "cpu", "--out", str(out)])  # fmt: skip
    df = pd.read_csv(out)
    assert set(df.columns) == RESULT_COLUMNS
    meta = json.loads((tmp_path / "probes_meta.json").read_text())
    assert meta["metadata_column"] is True and meta["num_variables"] == D
