"""Does a context stay inside what the model saw in training?

The integration fills `TrainingSupport` from the model's training
configuration; for CITNP (`integrations/citnp/support.py`), from the run's data
config (`MixedRegimeConfig`, see `sample_design` in
CITNP/datasets/mixed_regime_generator.py):

- total context rows: drawn from `n_ctx_range` (inclusive);
- regimes: an observational regime (always present when n_obs_min > 0, with
  at least n_obs_min + 1 rows: one row plus the reserve) and K <=
  min(k_max, D) single-node interventional regimes on DISTINCT nodes; the
  rows of one interventional regime share the target node and each has its
  own do-value;
- rows per regime: every regime gets one row, the rest is split by a
  Dirichlet-multinomial, so any regime size >= 1 occurs (single-row regimes
  included, though rarely when the Dirichlet concentration is large).

In the loop's terms, all experiments on node i form one interventional
regime, so the number of regimes is the number of distinct intervened nodes.
Violations give a warning and a flag, never an error.
"""

from dataclasses import dataclass

from causal_active_learning.context import ExperimentContext


@dataclass(frozen=True)
class TrainingSupport:
    """The context shapes of the training prior.

    Attributes:
    ----------
    context_size_range : tuple[int, int]
        n_ctx_range, inclusive.

    minimum_observational_rows : int
        Fewest observational rows of a training context.

    maximum_interventional_regimes : int
        min(k_max, D).
    """

    context_size_range: tuple[int, int]
    minimum_observational_rows: int
    maximum_interventional_regimes: int


def context_violations(
    context: ExperimentContext, support: TrainingSupport
) -> list[str]:
    """Ways in which `context` lies outside `support` (empty if inside)."""
    problems = []
    low, high = support.context_size_range
    if not low <= context.num_rows <= high:
        problems.append(
            f"context has {context.num_rows} rows, outside the training "
            f"n_ctx_range [{low}, {high}]"
        )
    observational = int((~context.intervened.any(-1)).sum())
    if observational < support.minimum_observational_rows:
        problems.append(
            f"context has {observational} observational rows, fewer than the "
            f"training minimum {support.minimum_observational_rows}"
        )
    regimes = int(context.intervened.any(0).sum())
    if regimes > support.maximum_interventional_regimes:
        problems.append(
            f"context has {regimes} interventional regimes (distinct "
            f"intervened nodes), more than the training maximum "
            f"{support.maximum_interventional_regimes}"
        )
    if bool((context.intervened.sum(-1) > 1).any()):
        problems.append("context has rows intervened on more than one node")
    return problems


def worst_case_violations(
    initial_context: ExperimentContext,
    experiment_budget: int,
    rows_per_experiment: int,
    num_target_nodes: int,
    support: TrainingSupport,
) -> list[str]:
    """Violations that the final context can reach: N0 + budget * R rows
    and up to min(budget, number of possible target nodes) new regimes."""
    problems = []
    final_rows = (
        initial_context.num_rows + experiment_budget * rows_per_experiment
    )
    low, high = support.context_size_range
    if final_rows > high:
        problems.append(
            f"the final context has {final_rows} rows, above the training "
            f"n_ctx_range maximum {high}"
        )
    if initial_context.num_rows < low:
        problems.append(
            f"the initial context has {initial_context.num_rows} rows, below "
            f"the training n_ctx_range minimum {low}"
        )
    existing = initial_context.intervened.any(0)
    regimes = int(existing.sum()) + min(experiment_budget, num_target_nodes)
    regimes = min(regimes, int(existing.numel()))
    if regimes > support.maximum_interventional_regimes:
        problems.append(
            f"up to {regimes} interventional regimes, above the training "
            f"maximum {support.maximum_interventional_regimes}"
        )
    observational = int((~initial_context.intervened.any(-1)).sum())
    if observational < support.minimum_observational_rows:
        problems.append(
            f"the initial context has {observational} observational rows, "
            f"fewer than the training minimum "
            f"{support.minimum_observational_rows}"
        )
    return problems
