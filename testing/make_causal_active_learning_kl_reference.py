"""Record the loop's tables under the KL objective BEFORE the entropy
objectives were added, for the backward-compatibility test.

Run once, from the repository root, on the commit before the change (b41ca76,
the last commit of feature/active-learning without
`information_gain_objective`), e.g. through a worktree:

    git worktree add /tmp/al_before b41ca76
    cp testing/make_causal_active_learning_kl_reference.py \\
        testing/causal_active_learning_stubs.py /tmp/al_before/testing/
    (cd /tmp/al_before && PYTHONPATH=. python \\
        testing/make_causal_active_learning_kl_reference.py \\
        --out $OLDPWD/testing/golden/causal_active_learning_kl_divergence.json)
    git worktree remove --force /tmp/al_before

`reference_tables` is also what the test calls with the current code and
`information_gain_objective="kl_divergence"`; the stub runs are deterministic
on a CPU with one torch thread (the thread count changes the order of
floating-point reductions, so it is fixed to 1 here), so the floats must
agree bit for bit. JSON keeps every float exactly (Python writes the
shortest repr that round-trips).
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_active_learning_stubs import (
    ConjugateStub,
    StubEnvironment,
    cheap_config,
    empty_context,
    query,
)

# (label, strategy, extra config settings, dense diagnostic at step 1)
RUNS = (
    ("eig", "expected_information_gain", {}, True),
    ("eig_grid", "expected_information_gain", {"value_search": "grid"}, False),
    (
        "eig_monte_carlo",
        "expected_information_gain",
        {"kl_estimator": "monte_carlo", "kl_monte_carlo_samples": 64},
        False,
    ),
    (
        "eig_shared",
        "expected_information_gain",
        {"fantasy_noise": "shared"},
        False,
    ),
    (
        "eig_costs",
        "expected_information_gain",
        {"observational_cost": 2.0, "intervention_cost": 0.5},
        False,
    ),
    ("observational", "observational", {}, False),
    ("random_fixed_value", "random_fixed_value", {}, False),
    ("random", "random", {}, False),
    ("treatment", "treatment", {}, False),
    ("exhaustive_true_gain", "exhaustive_true_gain", {}, False),
)
TABLE_NAMES = (
    "steps",
    "scores",
    "bayesian_optimisation_trace",
    "surrogate",
    "dense_value_diagnostic",
)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def reference_tables(**config_settings) -> dict[str, list[dict]]:
    """The rows of every table for every run of RUNS (two steps each, the
    conjugate stub on the stub environment), with `config_settings` added to
    every config if `ActiveLearningConfig` has those fields. Runs on one
    torch thread (restored afterwards)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _reference_tables(config_settings)
    finally:
        torch.set_num_threads(threads)


def _reference_tables(config_settings: dict) -> dict[str, list[dict]]:
    from causal_active_learning.config import ActiveLearningConfig
    from causal_active_learning.evaluation.records import step_rows
    from causal_active_learning.loop import (
        ActiveLearningGenerators,
        run_active_learning,
    )

    known = {f.name for f in fields(ActiveLearningConfig)}
    extra = {k: v for k, v in config_settings.items() if k in known}
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_NAMES}
    for seed, (label, strategy, settings, dense) in enumerate(RUNS):
        config = cheap_config(strategy=strategy, **settings, **extra)
        records = run_active_learning(
            ConjugateStub(),
            StubEnvironment(4),
            empty_context(D=4),
            query(),
            config,
            ActiveLearningGenerators.from_seed(seed),
            dense_value_diagnostic_steps=(1,) if dense else (),
            dense_value_diagnostic_points=5,
        )
        context_rows = 20
        for record in records:
            if record.real_rows is not None:
                context_rows += record.real_rows.shape[0]
            parts = step_rows(
                {"run": label, "strategy": strategy},
                record,
                context_rows,
                config.rows_per_acquisition,
                surrogate_steps=(1,),
                surrogate_grid_points=9,
            )
            for name, rows in parts.items():
                tables[name] += [
                    {k: _plain(v) for k, v in row.items()} for row in rows
                ]
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(reference_tables(), f, indent=0)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
