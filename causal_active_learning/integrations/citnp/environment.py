"""The true system for the CITNP integration (`ExperimentEnvironment`).

`LinearSCMEnvironment` wraps the evaluation code's `LinearSCM` (rows from `LinearSCM.sample`, the
truth from `LinearSCM.do_dist`). Environments come from the training prior
(`from_prior_task`, the task sampler of `kshot-prior`) or from a fixed
example (`from_example_model`, the model files of `kshot-example`).

The true SCM is sampled with a numpy Generator, because the existing sampler
(`LinearSCM.sample`) takes one; it is separate from every torch generator of
the loop, so acquisition settings never change the real data drawn for a
given seed and sequence of experiments.
"""

from dataclasses import dataclass

import numpy as np
import torch

from causal_active_learning.distributions import QueryMixture
from CITNP.evaluation.oracle import LinearSCM

# Central probability mass of a task's do-value distribution covered by the
# "prior_range" value domain.
PRIOR_RANGE_TAIL_PROBABILITY = 0.005


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
        return QueryMixture.gaussian(slope * psi, sd)

    def prior_value_range(self) -> tuple[float, float] | None:
        """The central 1 - 2 * PRIOR_RANGE_TAIL_PROBABILITY range of the
        task's do-value distribution (`do_dist_quantile`), or None."""
        if self.do_value_distribution is None:
            return None
        from CITNP.datasets.mixed_regime_generator import do_dist_quantile

        q = PRIOR_RANGE_TAIL_PROBABILITY
        low, high = do_dist_quantile(
            self.do_value_distribution, np.array([q, 1.0 - q])
        )
        return float(low), float(high)

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
