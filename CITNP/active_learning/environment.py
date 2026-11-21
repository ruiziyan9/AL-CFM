"""Where real experiments are run: the true SCM (plan section 6).

`ExperimentEnvironment` is the protocol the loop uses; `LinearSCMEnvironment`
wraps the evaluation code's `LinearSCM` (rows from `LinearSCM.sample`, the
truth from `LinearSCM.do_dist`). Environments come from the training prior
(`from_prior_task`, the task sampler of `kshot-prior`) or from a fixed
example (`from_example_model`, the model files of `kshot-example`).

The true SCM is sampled with a numpy Generator, because the existing sampler
(`LinearSCM.sample`) takes one; it is separate from every torch generator of
the loop, so acquisition settings never change the real data drawn for a
given seed and sequence of experiments.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from CITNP.active_learning.config import CandidateExperiment
from CITNP.evaluation.oracle import LinearSCM


class ExperimentEnvironment(Protocol):
    """A source of real data and of the true query distribution.

    Attributes:
    ----------
    num_variables : int
        D.

    do_value_distribution : dict | None
        Do-value distribution of the training prior for this environment
        (as in `sample_do_values`), used for the query grid and the
        "prior_range" value domain; None if unknown.
    """

    do_value_distribution: dict | None

    @property
    def num_variables(self) -> int:
        """D."""
        ...

    def sample_observational(
        self, num_rows: int, generator: np.random.Generator
    ) -> torch.Tensor:
        """[num_rows, D] float64 observational rows."""
        ...

    def run_experiment(
        self, candidate: CandidateExperiment, generator: np.random.Generator
    ) -> torch.Tensor:
        """[R, D] float64 rows of the experiment (column target_node exactly
        the do-value)."""
        ...

    def true_query_distribution(
        self, treatment: int, outcome: int, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean [V] and standard deviation [V] of the true
        p(X_outcome | do(X_treatment = psi)) at the psi values [V]."""
        ...


@dataclass
class LinearSCMEnvironment:
    """A linear Gaussian SCM as the environment.

    Attributes:
    ----------
    scm : LinearSCM
        x_j = sum_i B[i, j] x_i + e_j.

    do_value_distribution : dict | None
        The task's do-value distribution.

    graph : np.ndarray | None
        [D, D] adjacency, recorded for reporting.
    """

    scm: LinearSCM
    do_value_distribution: dict | None = None
    graph: np.ndarray | None = None

    @property
    def num_variables(self) -> int:
        """D."""
        return len(self.scm.s2)

    def sample_observational(self, num_rows, generator):
        x, _ = self.scm.sample(int(num_rows), None, None, generator)
        return torch.as_tensor(x, dtype=torch.float64)

    def run_experiment(self, candidate, generator):
        R = candidate.rows_per_experiment
        if candidate.target_node is None:
            return self.sample_observational(R, generator)
        assert candidate.intervention_value is not None
        values = np.full(R, float(candidate.intervention_value))
        x, _ = self.scm.sample(R, int(candidate.target_node), values, generator)
        return torch.as_tensor(x, dtype=torch.float64)

    def true_query_distribution(self, treatment, outcome, values):
        slope, variance = self.scm.do_dist(int(treatment), int(outcome))
        psi = torch.as_tensor(values, dtype=torch.float64)
        sd = torch.full_like(psi, float(np.sqrt(max(variance, 1e-12))))
        return slope * psi, sd

    @classmethod
    def from_prior_task(
        cls, data_config, task_index: int, base_seed: int, num_variables: int,
        fixed_graph: np.ndarray | None = None,
    ) -> "LinearSCMEnvironment":  # fmt: skip
        """Task `task_index` of the training prior (`tasks.prior_scm`, the
        sampler of `kshot-prior`; with `fixed_graph`, that DAG)."""
        from CITNP.evaluation.tasks import prior_scm

        scm, graph, do_dist, _ = prior_scm(
            data_config, task_index, base_seed, num_variables,
            fixed_graph=fixed_graph,
        )  # fmt: skip
        return cls(scm, do_dist, graph)

    @classmethod
    def from_example_model(cls, example) -> "LinearSCMEnvironment":
        """A fixed `ExampleModel` (`example_model.load_example_model` or
        `draw_example_model`)."""
        return cls(example.scm, dict(example.do_dist), example.graph)
