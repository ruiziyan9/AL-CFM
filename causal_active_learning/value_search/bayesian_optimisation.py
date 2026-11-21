"""Bayesian optimisation of the intervention value per node: GP surrogate and
the GP-UCB rule (ABCI Appendix D.7; Srinivas et al., 2010), all nodes in
lockstep."""

from dataclasses import dataclass, field

import torch

from causal_active_learning.value_search import (
    ValueEvaluation,
    ValueOptimisationResult,
    _draw_base_seed,
    _node_generator,
)
from causal_active_learning.value_search.domain import ValueDomain
from causal_active_learning.value_search.gaussian_process import (
    FittedGaussianProcess,
    fit_gaussian_process,
)


def initial_design(domain: ValueDomain, num_points: int) -> list[float]:
    """Deterministic, symmetric initial values: lower + width * j / (n + 1),
    j = 1..n (for n = 3 on [-c, c]: -c/2, 0, c/2)."""
    return [
        domain.lower + domain.width * (j + 1) / (num_points + 1)
        for j in range(num_points)
    ]


@dataclass
class _NodeState:
    domain: ValueDomain
    generator: torch.Generator
    evaluations: list[ValueEvaluation] = field(default_factory=list)

    def data(self):
        u = torch.tensor(
            [[e.standardised_value] for e in self.evaluations],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [e.expected_information_gain for e in self.evaluations],
            dtype=torch.float64,
        )
        se = torch.tensor(
            [e.standard_error for e in self.evaluations], dtype=torch.float64
        )
        return u, y, se**2


class BayesianOptimisationValueOptimiser:
    """GP-UCB per node, in lockstep across nodes (plan section 5.3-5.4).

    Per node: `initial_points` deterministic values (`initial_design`), then
    `iterations` rounds of: fit the GP (`fit_gaussian_process`, noise = the
    squared standard errors, `restarts` random starts), propose the maximiser
    of mu(u) + exploration_weight * sigma(u) over `acquisition_grid_points`
    values spanning the domain, evaluate. The recommendation is the
    evaluated value with the largest posterior mean of the final GP (not the
    largest noisy estimate, which is biased upward).

    Every round evaluates one value per node in one `evaluate` call. Each
    node's random numbers (GP restarts) come from a generator that depends
    only on one draw from `generator` and the node index, so optimising
    several nodes in lockstep gives the same results as optimising each alone
    with a copy of the same generator.
    """

    def __init__(
        self,
        initial_points: int = 3,
        iterations: int = 8,
        exploration_weight: float = 1.0,
        acquisition_grid_points: int = 512,
        restarts: int = 5,
    ):
        if initial_points < 2:
            raise ValueError("initial_points must be >= 2")
        self.initial_points = initial_points
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.acquisition_grid_points = acquisition_grid_points
        self.restarts = restarts

    def _fit(self, state: _NodeState) -> FittedGaussianProcess:
        u, y, noise = state.data()
        return fit_gaussian_process(
            u, y, noise, state.domain.lengthscale_bounds(), state.generator,
            restarts=self.restarts,
        )  # fmt: skip

    def _propose(self, state: _NodeState) -> ValueEvaluation:
        """The UCB maximiser on the acquisition grid (estimate left NaN)."""
        surrogate = self._fit(state)
        d = state.domain
        grid = torch.linspace(
            d.lower, d.upper, self.acquisition_grid_points, dtype=torch.float64
        )[:, None]
        mean, sd = surrogate.posterior(grid)
        ucb = mean + self.exploration_weight * sd
        best = int(torch.argmax(ucb))
        u = float(grid[best, 0])
        h = surrogate.hyperparameters
        return ValueEvaluation(
            round=-1, standardised_value=u, value=d.to_value(u),
            expected_information_gain=float("nan"),
            standard_error=float("nan"), acquisition_value=float(ucb[best]),
            lengthscale=h.lengthscale, output_scale=h.output_scale,
        )  # fmt: skip

    def optimise(self, evaluate, domains, generator):
        base_seed = _draw_base_seed(generator)
        states = {
            node: _NodeState(domain, _node_generator(base_seed, node))
            for node, domain in domains.items()
        }
        initial = {
            node: initial_design(s.domain, self.initial_points)
            for node, s in states.items()
        }
        total_rounds = self.initial_points + self.iterations
        for round_index in range(total_rounds):
            proposals: dict[int, ValueEvaluation] = {}
            for node, state in states.items():
                if round_index < self.initial_points:
                    u = initial[node][round_index]
                    proposals[node] = ValueEvaluation(
                        round=-1, standardised_value=u,
                        value=state.domain.to_value(u),
                        expected_information_gain=float("nan"),
                        standard_error=float("nan"),
                    )  # fmt: skip
                else:
                    proposals[node] = self._propose(state)
            nodes = list(proposals)
            estimates = evaluate([
                states[n].domain.candidate(proposals[n].standardised_value)
                for n in nodes
            ])  # fmt: skip
            for node, estimate in zip(nodes, estimates):
                p = proposals[node]
                p.round = round_index
                p.expected_information_gain = estimate.expected_information_gain
                p.standard_error = estimate.standard_error
                states[node].evaluations.append(p)
        results = {}
        for node, state in states.items():
            surrogate = self._fit(state)
            u, _, _ = state.data()
            mean, _ = surrogate.posterior(u)
            best = int(torch.argmax(mean))
            results[node] = ValueOptimisationResult(
                target_node=node,
                best_value=state.evaluations[best].value,
                best_standardised_value=state.evaluations[
                    best
                ].standardised_value,
                best_value_posterior_mean=float(mean[best]),
                evaluations=state.evaluations,
                domain=state.domain,
                surrogate=surrogate,
            )
        return results
