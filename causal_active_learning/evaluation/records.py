"""Step records to table rows (one CSV each). Model-agnostic."""

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from causal_active_learning.config import INFORMATION_GAIN_OBJECTIVES
from causal_active_learning.loop import ActiveLearningStep

TABLES = (
    "steps",
    "scores",
    "bayesian_optimisation_trace",
    "surrogate",
    "dense_value_diagnostic",
)


def node_label(node: int | None) -> str:
    """ "observational" for None, else the 0-based node index as text."""
    return "observational" if node is None else str(int(node))


@dataclass
class ActiveLearningTables:
    """Result tables of a run (see `step_rows` for the columns)."""

    steps: pd.DataFrame
    scores: pd.DataFrame
    bayesian_optimisation_trace: pd.DataFrame
    surrogate: pd.DataFrame
    dense_value_diagnostic: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in TABLES}


def objective_columns(
    values: dict[str, float],
    prefix: str = "",
    standard_errors: dict[str, float] | None = None,
) -> dict[str, float]:
    """One column per objective of INFORMATION_GAIN_OBJECTIVES,
    `<prefix><objective>` (NaN where missing), and with `standard_errors`
    also `<prefix><objective>_standard_error`."""
    out = {
        f"{prefix}{n}": values.get(n, np.nan) for n in INFORMATION_GAIN_OBJECTIVES
    }
    if standard_errors is not None:
        for n in INFORMATION_GAIN_OBJECTIVES:
            out[f"{prefix}{n}_standard_error"] = standard_errors.get(n, np.nan)
    return out


def step_rows(
    ident: dict,
    step: ActiveLearningStep,
    context_rows: int,
    rows_per_acquisition: int,
    surrogate_steps: Collection[int] = (1,),
    surrogate_grid_points: int = 512,
) -> dict[str, list[dict]]:
    """The table rows of one step record, each starting with `ident`
    (setting, task, restart, strategy, ...).

    - steps: step, number of variables, intervenable nodes, rows per
      acquisition, chosen node ("observational", or the 0-based node; empty at
      step 0), chosen value, choice clear, observational candidate available,
      the information gain objective, the predicted value of the selected
      objective (`predicted_information_gain`) and of every objective
      (`predicted_<objective>`), the realised information gain
      KL(after || before) and realised entropy reduction, query KL to truth,
      outside-support flag and reasons, newest regime size, context rows;
    - scores (EIG strategy): per candidate, node, value, re-estimated value
      of the selected objective (`expected_information_gain`) and its
      standard error, coherence gap, cost, surrogate mean and standard
      deviation, the objective, every objective's value and standard error
      (`<objective>`, `<objective>_standard_error`), drift and entropy shift;
    - bayesian_optimisation_trace: per BO evaluation, node, round, u, value,
      the selected objective's estimate, standard error, the UCB value that
      proposed it, lengthscale, output scale, nugget, noise mode, the
      objective and every objective's estimate;
    - surrogate: the final GP posterior per node on the acquisition grid, at
      `surrogate_steps`;
    - dense_value_diagnostic: per node, value and noise mode, the dense
      estimate of the selected objective and of every objective, and the
      distance (in u) between the BO recommendation and the dense grid's
      maximiser.
    """
    c = step.chosen
    at = {**ident, "step": step.step}
    out: dict[str, list[dict]] = {
        name: []
        for name in (
            "steps",
            "scores",
            "bayesian_optimisation_trace",
            "surrogate",
            "dense_value_diagnostic",
        )
    }
    out["steps"].append(
        {
            **at,
            "num_variables": step.num_variables,
            "intervenable_nodes": ",".join(map(str, step.intervenable_nodes)),
            "rows_per_acquisition": rows_per_acquisition,
            "chosen_node": "" if c is None else node_label(c.target_node),
            "chosen_value": np.nan
            if c is None or c.intervention_value is None
            else c.intervention_value,
            "choice_is_clear": step.choice_is_clear,
            "observational_candidate_available": (
                step.observational_candidate_available
            ),
            "information_gain_objective": step.information_gain_objective,
            "predicted_information_gain": step.predicted_information_gain,
            **objective_columns(step.predicted_objective_values, "predicted_"),
            "realised_information_gain": step.realised_information_gain,
            "realised_entropy_reduction": step.realised_entropy_reduction,
            "query_kl_to_truth": step.query_kl_to_truth,
            "outside_training_support": step.outside_training_support,
            "support_violations": "; ".join(step.support_violations or []),
            "newest_regime_size": step.newest_regime_size,
            "context_rows": context_rows,
        }
    )
    recommendations: dict[int, float] = {}
    if step.strategy == "expected_information_gain":
        for s in step.scores:
            result = s.value_search_result
            out["scores"].append(
                {
                    **at,
                    "node": node_label(s.acquisition.target_node),
                    "value": np.nan
                    if s.acquisition.intervention_value is None
                    else s.acquisition.intervention_value,
                    "expected_information_gain": s.expected_information_gain,
                    "standard_error": s.standard_error,
                    "coherence_gap": s.coherence_gap,
                    "cost": s.cost,
                    "surrogate_mean": np.nan
                    if result is None
                    else result.best_value_surrogate_mean,
                    "surrogate_standard_deviation": np.nan
                    if result is None
                    else result.best_value_surrogate_standard_deviation,
                    "is_chosen": s.acquisition == c,
                    "information_gain_objective": step.information_gain_objective,
                    **objective_columns(
                        s.objective_values,
                        standard_errors=s.objective_standard_errors,
                    ),
                    "drift": s.drift,
                    "entropy_shift": s.entropy_shift,
                }
            )
            if result is None:
                continue
            recommendations[result.target_node] = result.best_standardised_value
            if result.surrogate is None:
                continue
            for e in result.evaluations:
                out["bayesian_optimisation_trace"].append(
                    {
                        **at,
                        "node": result.target_node,
                        "round": e.round,
                        "standardised_value": e.standardised_value,
                        "value": e.value,
                        "expected_information_gain": e.expected_information_gain,
                        "standard_error": e.standard_error,
                        "acquisition_value": e.acquisition_value,
                        "lengthscale": e.lengthscale,
                        "output_scale": e.output_scale,
                        "nugget": e.nugget,
                        "noise_mode": e.noise_mode,
                        "information_gain_objective": (
                            step.information_gain_objective
                        ),
                        **objective_columns(e.objective_values),
                    }
                )
            if step.step not in surrogate_steps:
                continue
            d = result.domain
            grid = torch.linspace(
                d.lower, d.upper, surrogate_grid_points, dtype=torch.float64
            )
            mean, sd = result.surrogate.posterior(grid[:, None])
            for u, m_, s_ in zip(grid.tolist(), mean.tolist(), sd.tolist()):
                out["surrogate"].append(
                    {
                        **at,
                        "node": result.target_node,
                        "standardised_value": u,
                        "value": d.to_value(u),
                        "posterior_mean": m_,
                        "posterior_standard_deviation": s_,
                    }
                )
    for node, evaluations in (step.dense_value_diagnostic or {}).items():
        for mode in sorted({e.noise_mode for e in evaluations}):
            mine = [e for e in evaluations if e.noise_mode == mode]
            best = max(mine, key=lambda e: e.expected_information_gain)
            distance = (
                abs(best.standardised_value - recommendations[node])
                if node in recommendations
                else np.nan
            )
            for e in mine:
                out["dense_value_diagnostic"].append(
                    {
                        **at,
                        "node": node,
                        "noise_mode": mode,
                        "standardised_value": e.standardised_value,
                        "value": e.value,
                        "expected_information_gain": e.expected_information_gain,
                        "standard_error": e.standard_error,
                        "distance_to_recommendation": distance,
                        "information_gain_objective": (
                            step.information_gain_objective
                        ),
                        **objective_columns(e.objective_values),
                    }
                )
    return out
