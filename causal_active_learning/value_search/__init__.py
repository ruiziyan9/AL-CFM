"""Inner level of the bi-level optimisation: the most informative value of
every intervenable node (Bayesian optimisation by default, a grid as an
option). All nodes are searched in lockstep, one batched EIG evaluation per
round."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.value_search.domain import ValueDomain
from causal_active_learning.value_search.gaussian_process import (
    FittedGaussianProcess,
)

ExpectedInformationGainFunction = Callable[
    [list[CandidateExperiment]], list[InformationGainEstimate]
]


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


def make_value_optimiser(config) -> InterventionValueOptimiser:
    """The inner-level search named by `config.value_optimiser`."""
    from causal_active_learning.value_search.bayesian_optimisation import (
        BayesianOptimisationValueOptimiser,
    )
    from causal_active_learning.value_search.grid import GridValueOptimiser

    if config.value_optimiser == "grid":
        return GridValueOptimiser(config.grid_standardised_values)
    return BayesianOptimisationValueOptimiser(
        initial_points=config.bayesian_optimisation_initial_points,
        iterations=config.bayesian_optimisation_iterations,
        exploration_weight=config.ucb_exploration_weight,
        acquisition_grid_points=config.acquisition_grid_points,
        restarts=config.gaussian_process_restarts,
    )
