"""Mean and 95% confidence interval of the query KL over runs per step, and
merging of partial runs. Model-agnostic."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from causal_active_learning.evaluation.records import TABLES

SUMMARY_COLUMNS = [
    "setting",
    "strategy",
    "step",
    "num_runs",
    "mean_query_kl",
    "standard_error_query_kl",
    "ci95_lower",
    "ci95_upper",
    "fraction_outside_training_support",
    "fraction_observational_candidate_available",
]


def student_t_quantile(probability: float, degrees_of_freedom) -> np.ndarray:
    """Student t quantile (scipy, which the repository already uses)."""
    from scipy.stats import t

    return np.asarray(t.ppf(probability, degrees_of_freedom), dtype=float)


def summarise(steps: pd.DataFrame) -> pd.DataFrame:
    """One row per setting x strategy x step: number of runs (task x
    restart), mean query KL to truth, its standard error, the 95% CI of the
    mean with the Student t quantile on num_runs - 1 degrees of freedom (NaN
    with one run), the fraction of runs outside the training support and the
    fraction with the observational candidate available."""
    if steps.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    d = steps.copy()
    if "setting" not in d:
        d["setting"] = ""
    rows = []
    for (setting, strategy, step), g in d.groupby(
        ["setting", "strategy", "step"], sort=False
    ):
        kl = pd.to_numeric(g["query_kl_to_truth"], errors="coerce").dropna()
        n = len(kl)
        mean = float(kl.mean()) if n else np.nan
        se = float(kl.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        half = float(student_t_quantile(0.975, n - 1) * se) if n > 1 else np.nan
        rows.append(
            {
                "setting": setting,
                "strategy": strategy,
                "step": int(step),
                "num_runs": n,
                "mean_query_kl": mean,
                "standard_error_query_kl": se,
                "ci95_lower": mean - half,
                "ci95_upper": mean + half,
                "fraction_outside_training_support": float(
                    g["outside_training_support"].astype(bool).mean()
                ),
                "fraction_observational_candidate_available": float(
                    g["observational_candidate_available"].astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def read_table(path: str | Path) -> pd.DataFrame:
    """A CSV, or an empty frame if the file is missing or empty."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_runs(input_prefixes: Sequence[str], output_prefix: str) -> pd.DataFrame:
    """Concatenate the tables of partial runs (`<prefix>_<table>.csv`),
    drop exact duplicate rows, write them under `output_prefix` and
    recompute `<output_prefix>_summary.csv`. Returns the summary."""
    for name in TABLES:
        parts = [read_table(f"{p}_{name}.csv") for p in input_prefixes]
        parts = [p for p in parts if not p.empty]
        merged = (
            pd.concat(parts, ignore_index=True).drop_duplicates()
            if parts
            else pd.DataFrame()
        )
        merged.to_csv(f"{output_prefix}_{name}.csv", index=False)
    steps = read_table(f"{output_prefix}_steps.csv")
    summary = summarise(steps)
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    return summary
