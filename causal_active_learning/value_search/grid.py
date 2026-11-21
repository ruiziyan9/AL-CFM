"""Grid value search: every node evaluated at fixed values."""

from causal_active_learning.value_search import (
    ValueEvaluation,
    ValueSearchResult,
)


class GridValueSearch:
    """Evaluate a fixed set of values u in [-1, 1] per node (clipped,
    duplicates removed) and return the value with the largest estimate. All
    values of all nodes go into one `evaluate` call."""

    def __init__(self, standardised_values, noise_mode: str = "independent"):
        self.standardised_values = tuple(float(u) for u in standardised_values)
        self.noise_mode = noise_mode

    def search(self, evaluate, domains, generator):
        plan: list[tuple[int, float]] = []
        for node, domain in domains.items():
            us = sorted(
                {
                    min(max(u, domain.lower), domain.upper)
                    for u in self.standardised_values
                }
            )
            plan += [(node, u) for u in us]
        estimates = evaluate([domains[n].acquisition(u) for n, u in plan])
        results = {}
        for node, domain in domains.items():
            evaluations = [
                ValueEvaluation(
                    round=0,
                    standardised_value=u,
                    value=domain.to_value(u),
                    expected_information_gain=e.expected_information_gain,
                    standard_error=e.standard_error,
                    noise_mode=self.noise_mode,
                )
                for (n, u), e in zip(plan, estimates)
                if n == node
            ]
            best = max(evaluations, key=lambda e: e.expected_information_gain)
            results[node] = ValueSearchResult(
                target_node=node,
                best_value=best.value,
                best_standardised_value=best.standardised_value,
                best_value_surrogate_mean=best.expected_information_gain,
                evaluations=evaluations,
                domain=domain,
            )
        return results
