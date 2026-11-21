"""Summaries and merging (plan test 22; core)."""

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
