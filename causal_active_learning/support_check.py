"""Does a context stay inside what the model can take and was trained on?

`TrainingSupport` is filled by the integration from the model's training
configuration (for CITNP, `integrations/citnp/support.py`). A regime is "all
rows intervened on the same node": repeated acquisitions on a node join that
node's regime, as in the CITNP generator, where every row of a regime has its
own do-value.

Checks, at every step, so the step at which a run leaves the support is
recorded:
- D larger than the architecture can take: error before the loop starts;
- D not among the trained D values: warning and flag;
- context size outside the range, fewer observational rows than the minimum,
  more regimes than the maximum, do-values outside the range: warning and
  flag;
- the size of the newest regime is recorded (small regimes are rare in
  training; not a flag).

Known difference, not flagged: within a regime the training prior always has
distinct do-values, whereas an R-row acquisition repeats one do-value R
times.
"""

from dataclasses import dataclass


from causal_active_learning.context import ExperimentalContext


@dataclass(frozen=True)
class TrainingSupport:
    """What the model was trained on (None = unknown or unbounded).

    Attributes:
    ----------
    trained_num_variables : tuple[int, ...] | None
        The D values seen in training.

    maximum_num_variables : int | None
        The largest D the architecture can take (None: unbounded).

    context_size_range : tuple[int, int] | None
        Context rows, inclusive.

    minimum_observational_rows : int
        Fewest observational rows of a training context.

    maximum_interventional_regimes : int | None
        Most interventional regimes of a training context.

    rows_per_regime_range : tuple[int, int | None]
        Rows per interventional regime (upper None: unbounded).

    do_value_range : tuple[float, float] | None
        Raw do-value range of training interventions.

    observational_queries_trained : bool
        Whether the model was trained on queries without intervention.
    """

    trained_num_variables: tuple[int, ...] | None = None
    maximum_num_variables: int | None = None
    context_size_range: tuple[int, int] | None = None
    minimum_observational_rows: int = 0
    maximum_interventional_regimes: int | None = None
    rows_per_regime_range: tuple[int, int | None] = (1, None)
    do_value_range: tuple[float, float] | None = None
    observational_queries_trained: bool = False


def check_num_variables(
    num_variables: int, support: TrainingSupport
) -> list[str]:
    """Raise ValueError if D exceeds the architecture's maximum; return a
    violation if D was not trained on."""
    maximum = support.maximum_num_variables
    if maximum is not None and num_variables > maximum:
        raise ValueError(
            f"the context has D = {num_variables} variables, more than the "
            f"model's architecture can take ({maximum})"
        )
    trained = support.trained_num_variables
    if trained is not None and num_variables not in trained:
        return [
            f"D = {num_variables} was not trained on (trained D: {list(trained)})"
        ]
    return []


def number_of_regimes(context: ExperimentalContext) -> int:
    """Distinct intervened nodes (a regime = rows intervened on one node)."""
    return int(context.intervened.any(0).sum())


def context_violations(
    context: ExperimentalContext, support: TrainingSupport
) -> list[str]:
    """Ways in which `context` lies outside `support` (empty if inside)."""
    problems = check_num_variables(context.num_variables, support)
    if support.context_size_range is not None:
        low, high = support.context_size_range
        if not low <= context.num_rows <= high:
            problems.append(
                f"context has {context.num_rows} rows, outside the training "
                f"range [{low}, {high}]"
            )
    observational = int((~context.intervened.any(-1)).sum())
    if observational < support.minimum_observational_rows:
        problems.append(
            f"context has {observational} observational rows, fewer than the "
            f"training minimum {support.minimum_observational_rows}"
        )
    regimes = number_of_regimes(context)
    if (
        support.maximum_interventional_regimes is not None
        and regimes > support.maximum_interventional_regimes
    ):
        problems.append(
            f"context has {regimes} interventional regimes, more than the "
            f"training maximum {support.maximum_interventional_regimes}"
        )
    if support.do_value_range is not None:
        low_value, high_value = support.do_value_range
        values = context.values[context.intervened]
        outside = (values < low_value) | (values > high_value)
        if bool(outside.any()):
            problems.append(
                f"{int(outside.sum())} do-value(s) outside the training range "
                f"[{low_value:g}, {high_value:g}] (largest magnitude "
                f"{float(values[outside].abs().max()):.3g})"
            )
    return problems


def worst_case_violations(
    initial_context: ExperimentalContext,
    acquisition_budget: int,
    rows_per_acquisition: int,
    num_target_nodes: int,
    support: TrainingSupport,
) -> list[str]:
    """Violations the final context can reach: N0 + budget * R rows and up to
    min(budget, number of possible target nodes) new regimes."""
    problems = []
    final_rows = (
        initial_context.num_rows + acquisition_budget * rows_per_acquisition
    )
    if support.context_size_range is not None:
        low, high = support.context_size_range
        if final_rows > high:
            problems.append(
                f"the final context has {final_rows} rows, above the training "
                f"maximum {high}"
            )
        if initial_context.num_rows < low:
            problems.append(
                f"the initial context has {initial_context.num_rows} rows, "
                f"below the training minimum {low}"
            )
    if support.maximum_interventional_regimes is not None:
        existing = initial_context.intervened.any(0)
        regimes = int(existing.sum()) + min(acquisition_budget, num_target_nodes)
        regimes = min(regimes, int(existing.numel()))
        if regimes > support.maximum_interventional_regimes:
            problems.append(
                f"up to {regimes} interventional regimes, above the training "
                f"maximum {support.maximum_interventional_regimes}"
            )
    return problems
