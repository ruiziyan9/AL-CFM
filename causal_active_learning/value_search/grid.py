"""Grid value search: every node evaluated at fixed values."""

from causal_active_learning.value_search import (
    ValueEvaluation,
    ValueOptimisationResult,
)


class GridValueOptimiser:
    """Evaluate a fixed set of standardised values per node (clipped to the
    domain, duplicates removed) and return the best estimate. A baseline for
    BO and a tool for tests; all values of all nodes go into one call."""

    def __init__(self, standardised_values: tuple[float, ...]):
        self.standardised_values = tuple(float(u) for u in standardised_values)

    def optimise(self, evaluate, domains, generator):
        plan: list[tuple[int, float]] = []
        for node, domain in domains.items():
            us = sorted({
                min(max(u, domain.lower), domain.upper)
                for u in self.standardised_values
            })  # fmt: skip
            plan += [(node, u) for u in us]
        estimates = evaluate([domains[n].candidate(u) for n, u in plan])
        results = {}
        for node, domain in domains.items():
            evaluations = [
                ValueEvaluation(
                    round=0,
                    standardised_value=u,
                    value=domain.to_value(u),
                    expected_information_gain=e.expected_information_gain,
                    standard_error=e.standard_error,
                )  # fmt: skip
                for (n, u), e in zip(plan, estimates)
                if n == node
            ]
            best = max(evaluations, key=lambda e: e.expected_information_gain)
            results[node] = ValueOptimisationResult(
                target_node=node,
                best_value=best.value,
                best_standardised_value=best.standardised_value,
                best_value_posterior_mean=best.expected_information_gain,
                evaluations=evaluations,
                domain=domain,
            )
        return results
