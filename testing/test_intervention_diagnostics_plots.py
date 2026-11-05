"""Tests for the intervention-diagnostics figures (CITNP/evaluation/plots.py,
`plot-intervention-diagnostics`): every figure is built from tiny synthetic
CSVs with the columns that probe-intervention and context-edits write."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CITNP.evaluation.plots import (
    CONTEXT_EDITS_SUFFIXES,
    FIGURE_FORMATS,
    PROBES_SUFFIX,
    SENSITIVITY_RATIO_CSV,
    edit_sensitivity_table,
)

FIGURES = [
    "probes_by_layer",
    "edit_shift_scatter_mean_shift",
    "edit_shift_scatter_kl",
    "edit_shift_curve",
    "edit_sensitivity_ratio",
    "normalisation_effect",
    "secondary_discrepancy",
]
POINTS = ["embed", "block0/sample", "block0/node", "block1/sample", "block1/node"]
D = 3


def synthetic_probes(with_metadata: bool, seed: int) -> pd.DataFrame:
    """Rows as written by run_intervention_probes."""
    rng = np.random.default_rng(seed)
    probes = ["cell_row_intervened", "cell_row_intervened_excluding_targets",
              "cell_is_intervention_target"]  # fmt: skip
    if with_metadata:
        probes += ["metadata_row_intervened", "metadata_intervention_target"]
    rows = []
    for probe in probes:
        classes = D if probe == "metadata_intervention_target" else 2
        points = [(p, d) for d, p in enumerate(POINTS)]
        if probe.startswith("cell"):
            points = [("value_baseline", -1)] + points
        for point, depth in points:
            for labels in ("true", "shuffled"):
                score = rng.uniform(0.6, 1.0) if labels == "true" else 1 / classes
                rows.append({
                    "probe": probe, "point": point, "depth": depth,
                    "labels": labels, "accuracy": score,
                    "balanced_accuracy": score, "auc": np.nan,
                    "majority_class_rate": 0.5, "n": 100,
                    "num_classes": classes, "num_groups": 6,
                })  # fmt: skip
    return pd.DataFrame(rows)


def synthetic_edits(seed: int, num_tasks: int = 6) -> pd.DataFrame:
    """Per-row context-edit results with the columns of run_context_edits."""
    rng = np.random.default_rng(seed)
    rows = []
    for task in range(num_tasks):
        num_intervened = int(rng.integers(3, 12))
        counts, n = [], 1
        while n < num_intervened:
            counts.append(n)
            n *= 2
        counts.append(num_intervened)
        for T in range(D):
            for Y in range(D):
                if T == Y:
                    continue
                identifiable = bool(rng.random() < 0.5)
                treated = bool(rng.random() < 0.5)
                secondary_original = rng.gamma(2.0, 0.05)
                for edit in ("relabel_observational", "move_target"):
                    for n in counts:
                        oracle_kl = rng.gamma(2.0, 0.1) * n
                        model_kl = oracle_kl * rng.uniform(0.2, 1.5)
                        role = "not_applicable"
                        if edit == "move_target":
                            role = "mixed"  # several moved rows
                            if n == 1:
                                role = ("treatment", "outcome",
                                        "other")[rng.integers(3)]  # fmt: skip
                        oracle_shift = rng.normal(0, 0.3)
                        rows.append({
                            "D": D, "task": task, "T": T, "Y": Y,
                            "edit": edit, "num_edited_rows": n,
                            "num_intervened_rows": num_intervened,
                            "fraction_edited_rows": n / num_intervened,
                            "num_edited_rows_targeting_treatment": 0,
                            "relation": "unrelated",
                            "identifiable_from_observational": identifiable,
                            "treatment_intervened_in_context": treated,
                            "outcome_intervened_in_context": False,
                            "num_query_points": 8,
                            "new_target_role": role,
                            "num_moved_to_treatment": 0,
                            "num_moved_to_outcome": 0,
                            "num_moved_to_other": 0,
                            "model_mean_shift": oracle_shift * 0.7,
                            "model_mean_shift_rms": abs(oracle_shift) * 0.7,
                            "model_slope_change": oracle_shift * 0.5,
                            "oracle_mean_shift": oracle_shift,
                            "oracle_mean_shift_rms": abs(oracle_shift),
                            "oracle_slope_change": oracle_shift * 0.6,
                            "model_kl_original_to_edited": model_kl,
                            "model_kl_original_to_edited_se": 0.02,
                            "oracle_kl_original_to_edited": oracle_kl,
                            # some rows unresolved: KL < 3 standard errors
                            "oracle_kl_original_to_edited_se":
                                oracle_kl if task == 0 else 0.01,
                            "mean_shift_sign_agreement": 0.8,
                            "secondary_kl_oracle_to_model_original":
                                secondary_original,
                            "secondary_kl_oracle_to_model_original_se": 0.01,
                            "secondary_kl_oracle_to_model_edited":
                                rng.gamma(2.0, 0.2),
                            "secondary_kl_oracle_to_model_edited_se": 0.01,
                        })  # fmt: skip
    return pd.DataFrame(rows)


def write_results(directory: Path, label: str, seed: int) -> str:
    prefix = str(directory / label)
    synthetic_probes(label == "extra_node", seed).to_csv(
        prefix + PROBES_SUFFIX, index=False
    )
    for k, suffix in enumerate(CONTEXT_EDITS_SUFFIXES.values()):
        synthetic_edits(seed + k).to_csv(prefix + suffix, index=False)
    return prefix


def test_every_figure_from_synthetic_results(tmp_path):
    from CITNP.evaluation.__main__ import main

    results = tmp_path / "results"
    results.mkdir()
    prefixes = {
        label: write_results(results, label, seed)
        for seed, label in enumerate(["extra_node", "layerwise"])
    }
    out_dir = tmp_path / "figures"
    main(["plot-intervention-diagnostics",
          *[f"{label}={prefix}" for label, prefix in prefixes.items()],
          "--out_dir", str(out_dir), "--n_bootstrap", "50"])  # fmt: skip
    for figure in FIGURES:
        for fmt in FIGURE_FORMATS:
            path = out_dir / f"{figure}.{fmt}"
            assert path.stat().st_size > 0, path
    table = pd.read_csv(out_dir / SENSITIVITY_RATIO_CSV)
    assert set(table.columns) == {
        "model", "metric", "edit", "grouping",
        "identifiable_from_observational", "treatment_intervened_in_context",
        "new_target_role", "num_rows", "num_excluded", "num_tasks",
        "median_ratio", "ci_low", "ci_high",
    }  # fmt: skip
    assert set(table.model) == set(prefixes)
    roles = table[table.grouping != "stratum"]
    assert set(roles.edit) == {"move_target"}
    assert set(roles.new_target_role) == {"treatment", "outcome", "other"}


def test_sensitivity_exclusion_and_bootstrap():
    df = synthetic_edits(0)
    table = edit_sensitivity_table(
        {"m": df}, min_kl_standard_errors=3.0, n_bootstrap=50, seed=1
    )
    overall = table[(table.metric == "kl") & (table.grouping == "stratum")
                    & (table.identifiable_from_observational == "all")]  # fmt: skip
    for row in overall.itertuples():
        rows = df[df.edit == row.edit]
        unresolved = rows.oracle_kl_original_to_edited < (
            3.0 * rows.oracle_kl_original_to_edited_se
        )
        assert row.num_rows == len(rows)
        assert row.num_excluded == int(unresolved.sum()) > 0  # task 0
        assert row.num_tasks == rows.task.nunique() - 1
        assert row.ci_low <= row.ci_high
    again = edit_sensitivity_table(
        {"m": df}, min_kl_standard_errors=3.0, n_bootstrap=50, seed=1
    )
    pd.testing.assert_frame_equal(table, again)
    looser = edit_sensitivity_table(
        {"m": df}, min_kl_standard_errors=0.5, n_bootstrap=50, seed=1
    )
    assert looser.num_excluded.sum() < table.num_excluded.sum()


def test_missing_files_are_skipped(tmp_path):
    from CITNP.evaluation.plots import plot_intervention_diagnostics

    prefix = str(tmp_path / "probes_only")
    synthetic_probes(True, 0).to_csv(prefix + PROBES_SUFFIX, index=False)
    written = plot_intervention_diagnostics(
        {"extra_node": prefix}, tmp_path / "figures", n_bootstrap=10
    )
    assert {p.stem for p in written} == {"probes_by_layer"}


def test_bad_model_argument_is_rejected(tmp_path):
    from CITNP.evaluation.__main__ import main

    with pytest.raises(SystemExit):
        main(["plot-intervention-diagnostics", "no_equals_sign",
              "--out_dir", str(tmp_path)])  # fmt: skip
