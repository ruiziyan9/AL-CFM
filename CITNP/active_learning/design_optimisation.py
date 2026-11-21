"""Bi-level optimisation of the next experiment (plan section 5; ABCI
Eq. 4.15):

    i* in argmax_i EIG(i, v_i*),   where for every i:  v_i* in argmax_v EIG(i, v).

Inner level (`InterventionValueOptimiser`): for every candidate node, the
most informative value, by Bayesian optimisation (BO) with a GP surrogate and
the GP-UCB rule (ABCI Appendix D.7; Srinivas et al., 2010), or on a fixed
grid. All nodes are optimised in lockstep, so each round is one batched EIG
evaluation.

Outer level (`reevaluate_and_select`): every node's recommended experiment is
re-evaluated with more, fresh fantasies and the best is chosen, which removes
the upward bias of picking the maximum of noisy inner-level estimates.

Values are optimised in a standardised coordinate u, v = centre + scale * u
(`ValueDomain`), so the GP lengthscale is comparable across nodes and tasks.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from CITNP.active_learning.config import ActiveLearningConfig, CandidateExperiment
from CITNP.active_learning.gaussian_process import (
    FittedGaussianProcess,
    fit_gaussian_process,
)
from CITNP.active_learning.information_gain import InformationGainEstimate

# The GP lengthscale is bounded to [0.05, 2] x the domain width.
LENGTHSCALE_BOUNDS_IN_DOMAIN_WIDTHS = (0.05, 2.0)

ExpectedInformationGainFunction = Callable[
    [list[CandidateExperiment]], list[InformationGainEstimate]
]


@dataclass(frozen=True)
class ValueDomain:
    """The interval of values searched for one node, in the standardised
    coordinate u, with v = centre + scale * u on the raw scale.

    Attributes:
    ----------
    target_node : int
        The node.

    lower, upper : float
        Bounds of u.

    centre, scale : float
        Raw-scale mean and standard deviation of the node in the context
        (the model's standardisation).

    rows_per_experiment : int
        Rows of every candidate built from this domain.
    """

    target_node: int
    lower: float
    upper: float
    centre: float
    scale: float
    rows_per_experiment: int = 1

    def __post_init__(self):
        if not self.lower < self.upper:
            raise ValueError("a value domain needs lower < upper")
        if not self.scale > 0:
            raise ValueError("a value domain needs scale > 0")

    @property
    def width(self) -> float:
        """upper - lower."""
        return self.upper - self.lower

    def to_value(self, standardised_value: float) -> float:
        """u -> v (raw scale)."""
        return self.centre + self.scale * float(standardised_value)

    def to_standardised(self, value: float) -> float:
        """v -> u."""
        return (float(value) - self.centre) / self.scale

    def candidate(self, standardised_value: float) -> CandidateExperiment:
        """The experiment do(X_node = to_value(u))."""
        return CandidateExperiment(
            self.target_node,
            self.to_value(standardised_value),
            self.rows_per_experiment,
        )

    def lengthscale_bounds(self) -> tuple[float, float]:
        """GP lengthscale bounds, LENGTHSCALE_BOUNDS_IN_DOMAIN_WIDTHS x width."""
        low, high = LENGTHSCALE_BOUNDS_IN_DOMAIN_WIDTHS
        return low * self.width, high * self.width


@dataclass
class ValueEvaluation:
    """One EIG evaluation of the inner level.

    Attributes:
    ----------
    round : int
        Lockstep round (0-based; initial points first).

    standardised_value, value : float
        u and v.

    expected_information_gain, standard_error : float
        The estimate.

    acquisition_value : float
        Upper confidence bound that proposed the value (NaN for the initial
        design and the grid).

    lengthscale, output_scale : float
        Hyperparameters of the surrogate that proposed it, on the
        standardised target scale (NaN for the initial design and the grid).
    """

    round: int
    standardised_value: float
    value: float
    expected_information_gain: float
    standard_error: float
    acquisition_value: float = float("nan")
    lengthscale: float = float("nan")
    output_scale: float = float("nan")


@dataclass
class ValueOptimisationResult:
    """Inner-level result for one node.

    Attributes:
    ----------
    target_node : int
        The node.

    best_value, best_standardised_value : float
        The recommended v and u.

    best_value_posterior_mean : float
        Surrogate estimate of EIG at the recommended value (BO: GP posterior
        mean; grid: the estimate itself).

    evaluations : list[ValueEvaluation]
        Every evaluation, in order.

    domain : ValueDomain
        The searched domain.

    surrogate : FittedGaussianProcess | None
        BO only: the final GP (fitted to all evaluations).
    """

    target_node: int
    best_value: float
    best_standardised_value: float
    best_value_posterior_mean: float
    evaluations: list[ValueEvaluation]
    domain: ValueDomain
    surrogate: FittedGaussianProcess | None = None

    @property
    def candidate(self) -> CandidateExperiment:
        """The recommended experiment."""
        return self.domain.candidate(self.best_standardised_value)


class InterventionValueOptimiser(Protocol):
    """Inner level: the most informative value for every node."""

    def optimise(
        self,
        evaluate: ExpectedInformationGainFunction,
        domains: dict[int, ValueDomain],
        generator: torch.Generator,
    ) -> dict[int, ValueOptimisationResult]:
        """Optimise all nodes of `domains`; `evaluate` maps candidates to
        `InformationGainEstimate`s (one batched call per round)."""
        ...


def _node_generator(base_seed: int, node: int) -> torch.Generator:
    """A generator that depends only on (base_seed, node)."""
    state = np.random.SeedSequence([base_seed, node]).generate_state(2)
    seed = (int(state[0]) << 32 | int(state[1])) & ((1 << 63) - 1)
    return torch.Generator().manual_seed(seed)


def _draw_base_seed(generator: torch.Generator) -> int:
    return int(torch.randint(0, 2**62, (1,), generator=generator))


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


def make_value_optimiser(
    config: ActiveLearningConfig,
) -> InterventionValueOptimiser:
    """The inner-level optimiser named by `config.value_optimiser`."""
    if config.value_optimiser == "grid":
        return GridValueOptimiser(config.grid_standardised_values)
    return BayesianOptimisationValueOptimiser(
        initial_points=config.bayesian_optimisation_initial_points,
        iterations=config.bayesian_optimisation_iterations,
        exploration_weight=config.ucb_exploration_weight,
        acquisition_grid_points=config.acquisition_grid_points,
        restarts=config.gaussian_process_restarts,
    )


# --------------------------------------------------------------------------- #
# Outer level
# --------------------------------------------------------------------------- #
@dataclass
class CandidateScore:
    """Outer-level result for one candidate.

    Attributes:
    ----------
    candidate : CandidateExperiment
        The node's recommended experiment.

    expected_information_gain, standard_error : float
        The re-evaluated estimate (fresh, `final_num_fantasies` fantasies).

    coherence_gap : float
        See `InformationGainEstimate.coherence_gap`.

    inner_result : ValueOptimisationResult | None
        The inner-level result (None for the observational candidate).
    """

    candidate: CandidateExperiment
    expected_information_gain: float
    standard_error: float
    coherence_gap: float
    inner_result: ValueOptimisationResult | None


def choice_is_clear(scores: list[CandidateScore]) -> bool:
    """False if the runner-up is within two combined standard errors of the
    best, sqrt(se_best^2 + se_second^2); True with a single candidate."""
    if len(scores) < 2:
        return True
    ranked = sorted(scores, key=lambda s: s.expected_information_gain)
    best, second = ranked[-1], ranked[-2]
    combined = math.sqrt(best.standard_error**2 + second.standard_error**2)
    gap = best.expected_information_gain - second.expected_information_gain
    return bool(gap > 2.0 * combined)


def reevaluate_and_select(
    evaluate_final: ExpectedInformationGainFunction,
    inner_results: dict[int, ValueOptimisationResult],
    extra_candidates: list[CandidateExperiment] | None = None,
) -> tuple[CandidateExperiment, list[CandidateScore]]:
    """Outer level: re-evaluate every node's recommended experiment (and the
    `extra_candidates`, e.g. an observational experiment) with
    `evaluate_final` (fresh noise, `final_num_fantasies`), and return the
    candidate with the largest re-evaluated EIG and all scores (in node
    order, extra candidates last)."""
    nodes = list(inner_results)
    candidates = [inner_results[n].candidate for n in nodes]
    candidates += list(extra_candidates or [])
    estimates = evaluate_final(candidates)
    scores = [
        CandidateScore(
            candidate=e.candidate,
            expected_information_gain=e.expected_information_gain,
            standard_error=e.standard_error,
            coherence_gap=e.coherence_gap,
            inner_result=inner_results[nodes[i]] if i < len(nodes) else None,
        )
        for i, e in enumerate(estimates)
    ]
    best = max(scores, key=lambda s: s.expected_information_gain)
    return best.candidate, scores
