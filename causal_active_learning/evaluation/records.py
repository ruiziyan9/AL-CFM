"""Step records to tables (one CSV each). Model-agnostic."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from causal_active_learning.loop import ActiveLearningStep

# Points of the saved GP posterior per node and step.
SURROGATE_GRID_POINTS = 101


@dataclass
class ActiveLearningTables:
    """Results as tables (one CSV each).

    Attributes:
    ----------
    steps : pd.DataFrame
        One row per task x strategy x seed x step.

    scores : pd.DataFrame
        One row per task x strategy x seed x step x candidate node (EIG
        strategies only).

    trace : pd.DataFrame
        One row per BO evaluation (EIG strategies with BO only).

    surrogate : pd.DataFrame
        The final GP posterior of every node on a grid (BO only).

    dense : pd.DataFrame
        Dense value diagnostic (when requested).
    """

    steps: pd.DataFrame
    scores: pd.DataFrame
    trace: pd.DataFrame
    surrogate: pd.DataFrame
    dense: pd.DataFrame


def step_rows(ident: dict, step: ActiveLearningStep, context_rows: int):
    """The table rows of one step record: (steps row, scores rows, trace
    rows, surrogate rows, dense diagnostic rows), each with `ident`."""
    c = step.chosen
    steps = {
        **ident, "step": step.step,
        "chosen_node": -1 if c.target_node is None else c.target_node,
        "chosen_value": np.nan if c.intervention_value is None
        else c.intervention_value,
        "chosen": c.describe(),
        "choice_is_clear": step.choice_is_clear,
        "predicted_information_gain": step.predicted_information_gain,
        "realised_information_gain": step.realised_information_gain,
        "kl_to_truth_before": step.kl_to_truth_before,
        "kl_to_truth_after": step.kl_to_truth_after,
        "context_rows": context_rows,
        "outside_training_support": step.outside_training_support,
        "support_violations": "; ".join(step.support_violations),
    }  # fmt: skip
    scores, trace, surrogate, dense = [], [], [], []
    at = {**ident, "step": step.step}
    if step.strategy == "expected_information_gain":
        for s in step.scores:
            inner = s.value_search_result
            scores.append({
                **at,
                "candidate_node": -1 if s.acquisition.target_node is None
                else s.acquisition.target_node,
                "optimised_value": np.nan
                if s.acquisition.intervention_value is None
                else s.acquisition.intervention_value,
                "optimised_standardised_value": np.nan if inner is None
                else inner.best_standardised_value,
                "expected_information_gain": s.expected_information_gain,
                "standard_error": s.standard_error,
                "coherence_gap": s.coherence_gap,
                "surrogate_estimate": np.nan if inner is None
                else inner.best_value_surrogate_mean,
                "is_chosen": s.acquisition == c,
            })  # fmt: skip
            if inner is None or inner.surrogate is None:
                continue
            for index, e in enumerate(inner.evaluations):
                trace.append({
                    **at, "node": inner.target_node, "evaluation": index,
                    "round": e.round, "standardised_value": e.standardised_value,
                    "value": e.value,
                    "expected_information_gain": e.expected_information_gain,
                    "standard_error": e.standard_error,
                    "acquisition_value": e.acquisition_value,
                    "lengthscale": e.lengthscale,
                    "output_scale": e.output_scale,
                })  # fmt: skip
            d = inner.domain
            grid = torch.linspace(
                d.lower, d.upper, SURROGATE_GRID_POINTS, dtype=torch.float64
            )
            mean, sd = inner.surrogate.posterior(grid[:, None])
            for u, m_, s_ in zip(grid.tolist(), mean.tolist(), sd.tolist()):
                surrogate.append({
                    **at, "node": inner.target_node, "standardised_value": u,
                    "value": d.to_value(u), "posterior_mean": m_,
                    "posterior_standard_deviation": s_,
                })  # fmt: skip
    for node, evaluations in (step.dense_value_diagnostic or {}).items():
        for e in evaluations:
            dense.append({
                **at, "node": node, "standardised_value": e.standardised_value,
                "value": e.value,
                "expected_information_gain": e.expected_information_gain,
                "standard_error": e.standard_error,
            })  # fmt: skip
    return steps, scores, trace, surrogate, dense
