"""What the active learning loop needs from a model and from the system
experiments are run on (Protocol classes only, no logic).

To use another causal foundation model, implement `PredictiveModel`; to use
another data source, implement `ExperimentEnvironment`.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.context import ExperimentContext
from causal_active_learning.distributions import QueryMixture


class PredictiveModel(Protocol):
    """A predictive causal model.

    Fantasies (the model's imagined rows of an acquisition) must be drawn by
    plain sampling from the model's predictive distribution, ROW BY ROW: row 1
    from p(X | a, D) (for an intervention, the joint prediction under
    do(X_i = v) with column i set to v exactly), row 2 from
    p(X | a, D + row 1), and so on. This autoregressive order is the exact
    predictive chain rule: the rows of one experiment share the same unknown
    mechanism and each row informs the next. Drawing the rows independently
    from one prediction overstates how much a multi-row experiment reveals.

    The model never draws random numbers on its own: the standard noise a
    fantasy is computed from comes from `draw_fantasy_noise` and is passed
    back, so the caller decides when noise is fresh and when it is shared.

    Attributes:
    ----------
    supports_observational_prediction : bool
        Whether the model can predict a new observational row p(X | D)
        (needed for the EIG of the observational candidate).
    """

    supports_observational_prediction: bool

    def predict_query(
        self,
        contexts: Sequence[ExperimentContext],
        treatment: int,
        outcome: int,
        treatment_values: torch.Tensor,
    ) -> QueryMixture:
        """p(X_outcome | do(X_treatment = psi), context_b) for every context b
        (all of equal size) and psi in treatment_values [Q]: a mixture
        [B, Q, K], raw scale, float64."""
        ...

    def draw_fantasy_noise(
        self,
        num_variables: int,
        num_fantasies: int,
        num_rows: int,
        generator: torch.Generator,
    ) -> Any:
        """The standard noise of `num_fantasies` fantasies of `num_rows` rows
        each (opaque to the caller)."""
        ...

    def sample_fantasies(
        self,
        contexts: Sequence[ExperimentContext],
        acquisitions: Sequence[CandidateExperiment],
        noise: Sequence[Any],
    ) -> torch.Tensor:
        """Fantasies [B, S, R, D] of acquisition b on context b computed from
        noise b (S fantasies of R rows, raw scale, float64); interventional
        columns equal the do-value exactly."""
        ...


class ExperimentEnvironment(Protocol):
    """A source of real data and, for simulations, of the true query
    distribution.

    `generator` is whatever random generator the implementation uses (numpy
    for the CITNP integration); it is a separate stream from all acquisition
    randomness.
    """

    @property
    def num_variables(self) -> int:
        """D."""
        ...

    def sample_observational(self, num_rows: int, generator: Any) -> torch.Tensor:
        """[num_rows, D] float64 observational rows."""
        ...

    def run_experiment(
        self, candidate: CandidateExperiment, generator: Any
    ) -> torch.Tensor:
        """[R, D] float64 rows of the acquisition (R independent draws from the
        hard-intervened system at the do-value, target column exact; or R
        observational draws)."""
        ...

    def true_query_distribution(
        self, treatment: int, outcome: int, values: torch.Tensor
    ) -> QueryMixture:
        """The true p(X_outcome | do(X_treatment = psi)) at psi in values [Q]:
        a mixture [Q, K]."""
        ...
