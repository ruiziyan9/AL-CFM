"""Summaries and merging (plan test 22), records and plots of the
information gain objectives (entropy objective plan test 9); core."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from causal_active_learning.evaluation.summaries import merge_runs, summarise


def _steps(seed=0, tasks=4, restarts=3, steps=3):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(tasks):
        for r in range(restarts):
            for strategy in ("random", "expected_information_gain"):
                for k in range(steps):
                    rows.append(
                        {
                            "setting": "a",
                            "task": t,
                            "restart": r,
                            "strategy": strategy,
                            "step": k,
                            "query_kl_to_truth": float(rng.gamma(2.0)),
                            "outside_training_support": bool(rng.random() < 0.3),
                            "observational_candidate_available": strategy
                            == "random",
                        }
                    )
    return pd.DataFrame(rows)


def test_summary_matches_a_direct_computation():
    steps = _steps()
    summary = summarise(steps)
    row = summary[(summary.strategy == "random") & (summary.step == 1)].iloc[0]
    kl = steps[(steps.strategy == "random") & (steps.step == 1)].query_kl_to_truth
    n = len(kl)
    se = kl.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    assert row.num_runs == n == 12
    assert row.mean_query_kl == pytest.approx(kl.mean())
    assert row.standard_error_query_kl == pytest.approx(se)
    assert row.ci95_lower == pytest.approx(kl.mean() - half)
    assert row.ci95_upper == pytest.approx(kl.mean() + half)
    outside = steps[(steps.strategy == "random") & (steps.step == 1)]
    assert row.fraction_outside_training_support == pytest.approx(
        outside.outside_training_support.mean()
    )


def test_merging_partial_runs_gives_the_same_summary(tmp_path):
    steps = _steps()
    whole = summarise(steps)
    for i, part in enumerate((steps[steps.task < 2], steps[steps.task >= 2])):
        part.to_csv(tmp_path / f"part{i}_steps.csv", index=False)
    merged = merge_runs(
        [str(tmp_path / "part0"), str(tmp_path / "part1")], str(tmp_path / "all")
    )
    pd.testing.assert_frame_equal(
        merged.sort_values(["strategy", "step"]).reset_index(drop=True),
        whole.sort_values(["strategy", "step"]).reset_index(drop=True),
    )
    assert (tmp_path / "all_summary.csv").is_file()


# Entropy objective test 9: records of every objective
sys.path.insert(0, str(Path(__file__).resolve().parent))
OBJECTIVES = ("entropy_reduction", "kl_divergence", "mixture_entropy_reduction")


def _tables(objective, strategy="expected_information_gain"):
    from causal_active_learning_stubs import (
        ConjugateStub,
        StubEnvironment,
        cheap_config,
        empty_context,
        query,
    )

    from causal_active_learning.evaluation.records import step_rows
    from causal_active_learning.loop import (
        ActiveLearningGenerators,
        run_active_learning,
    )

    config = cheap_config(strategy=strategy, information_gain_objective=objective)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run_active_learning(
            ConjugateStub(),
            StubEnvironment(4),
            empty_context(D=4),
            query(),
            config,
            ActiveLearningGenerators.from_seed(1),
            dense_value_diagnostic_steps=(1,),
            dense_value_diagnostic_points=3,
        )
    tables: dict[str, list[dict]] = {}
    for record in records:
        for name, rows in step_rows(
            {"setting": "s", "task": 0, "restart": 0, "strategy": strategy},
            record,
            20,
            1,
        ).items():
            tables.setdefault(name, []).extend(rows)
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_records_hold_every_objective(objective):
    t = _tables(objective)
    steps = t["steps"]
    for column in (
        "information_gain_objective",
        "realised_entropy_reduction",
        *(f"predicted_{o}" for o in OBJECTIVES),
    ):
        assert column in steps, column
    assert (steps.information_gain_objective == objective).all()
    later = steps[steps.step > 0]
    np.testing.assert_array_equal(
        later.predicted_information_gain, later[f"predicted_{objective}"]
    )
    assert later.realised_entropy_reduction.notna().all()
    assert steps[steps.step == 0].realised_entropy_reduction.isna().all()
    scores = t["scores"]
    for column in (
        "drift",
        "entropy_shift",
        *OBJECTIVES,
        *(f"{o}_standard_error" for o in OBJECTIVES),
    ):
        assert scores[column].notna().all(), column
    np.testing.assert_array_equal(
        scores.expected_information_gain, scores[objective]
    )
    np.testing.assert_array_equal(
        scores.standard_error, scores[f"{objective}_standard_error"]
    )
    chosen = scores[scores.is_chosen]
    np.testing.assert_array_equal(
        chosen.expected_information_gain, later.predicted_information_gain
    )
    for name in ("bayesian_optimisation_trace", "dense_value_diagnostic"):
        table = t[name]
        assert (table.information_gain_objective == objective).all()
        assert table[list(OBJECTIVES)].notna().all().all()
        np.testing.assert_array_equal(
            table.expected_information_gain, table[objective]
        )


def test_baseline_records_have_empty_predictions():
    steps = _tables("entropy_reduction", strategy="random")["steps"]
    later = steps[steps.step > 0]
    assert later[[f"predicted_{o}" for o in OBJECTIVES]].isna().all().all()
    assert later.realised_entropy_reduction.notna().all()


def test_plots_with_negative_values_and_old_tables(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from causal_active_learning.evaluation.plots import (
        plot_active_learning_all,
        plot_calibration,
    )

    t = _tables("entropy_reduction")
    prefix = str(tmp_path / "run")
    for name, df in t.items():
        df.to_csv(f"{prefix}_{name}.csv", index=False)
    written = plot_active_learning_all(prefix, all_calibration_pairs=True)
    assert prefix + "_calibration.png" in written
    assert any("_surrogate_" in w for w in written)
    steps = t["steps"].copy()
    # negative predicted and realised values
    steps["predicted_entropy_reduction"] = -steps.predicted_entropy_reduction
    steps["realised_entropy_reduction"] = -steps.realised_entropy_reduction
    assert plot_calibration(steps, str(tmp_path / "negative"))
    # tables written before the objective columns existed
    old = steps[
        [
            "strategy",
            "step",
            "predicted_information_gain",
            "realised_information_gain",
        ]
    ]
    assert plot_calibration(old, str(tmp_path / "old"))
    no_predictions = old.assign(predicted_information_gain=np.nan)
    assert plot_calibration(no_predictions, str(tmp_path / "none")) is None
