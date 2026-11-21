"""Acquisitions, the query, and which acquisitions are allowed.

`build_candidate_set` is the only place that decides which acquisitions a
strategy may choose; `ConstrainedEnvironment` enforces the same constraint on
the system itself, so no strategy can bypass it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Acquisition:
    """A batch of R rows: R rows under one intervention do(X_i = v), all at
    the same value, or R observational rows.

    Attributes:
    ----------
    target_node : int | None
        0-based node set by do(); None for observational rows.

    intervention_value : float | None
        Do-value on the raw scale; None if and only if observational.

    num_rows : int
        R.
    """

    target_node: int | None
    intervention_value: float | None
    num_rows: int = 1

    def __post_init__(self):
        if (self.target_node is None) != (self.intervention_value is None):
            raise ValueError(
                "target_node and intervention_value must both be None "
                "(observational) or both be given"
            )
        if self.num_rows < 1:
            raise ValueError("num_rows must be >= 1")

    @classmethod
    def intervention(
        cls, node: int, value: float, num_rows: int = 1
    ) -> "Acquisition":
        """R = num_rows rows under do(X_node = value)."""
        return cls(int(node), float(value), int(num_rows))

    @classmethod
    def observational(cls, num_rows: int = 1) -> "Acquisition":
        """R = num_rows observational rows."""
        return cls(None, None, int(num_rows))

    @property
    def is_observational(self) -> bool:
        """True for observational rows."""
        return self.target_node is None

    def describe(self) -> str:
        """Readable name, e.g. "3 x do(X3 = 1.25)" (1-based variable names)."""
        if self.target_node is None:
            return f"{self.num_rows} x observational"
        return (
            f"{self.num_rows} x do(X{self.target_node + 1} = "
            f"{self.intervention_value:+.4g})"
        )


@dataclass(frozen=True)
class InterventionalQuery:
    """The target query p(X_outcome | do(X_treatment = psi)) at the psi
    values `treatment_values` [Q] (raw scale)."""

    treatment: int
    outcome: int
    treatment_values: torch.Tensor

    def __post_init__(self):
        psi = torch.as_tensor(self.treatment_values, dtype=torch.float64)
        if psi.ndim != 1 or len(psi) < 1:
            raise ValueError("treatment_values must be a non-empty 1-D tensor")
        if self.treatment == self.outcome:
            raise ValueError("treatment and outcome must differ")
        object.__setattr__(self, "treatment_values", psi)


@dataclass(frozen=True)
class CandidateSet:
    """The acquisitions a strategy may choose from: interventions on
    `intervenable_nodes` (at any value) and, if `observational`, an
    observational batch."""

    intervenable_nodes: tuple[int, ...]
    observational: bool


class NonIntervenableNodeError(ValueError):
    """An intervention on a node outside the intervenable set."""


def resolve_intervenable_nodes(
    num_variables: int, intervenable_nodes: Sequence[int] | None
) -> tuple[int, ...]:
    """The intervenable nodes (None = all D), validated: in range, unique."""
    if intervenable_nodes is None:
        return tuple(range(num_variables))
    nodes = tuple(int(n) for n in intervenable_nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"intervenable_nodes has duplicates: {nodes}")
    bad = [n for n in nodes if not 0 <= n < num_variables]
    if bad:
        raise ValueError(
            f"intervenable_nodes {bad} are outside 0..{num_variables - 1}"
        )
    return tuple(sorted(nodes))


def build_candidate_set(
    num_variables: int,
    intervenable_nodes: Sequence[int] | None,
    include_observational: bool,
    observational_available: bool,
) -> CandidateSet:
    """The candidate set of every strategy.

    Args:
    ----------
    num_variables : int
        D.

    intervenable_nodes : Sequence[int] | None
        None = all D nodes. The treatment and outcome are candidates if and
        only if listed.

    include_observational : bool
        `ActiveLearningConfig.include_observational_candidate`.

    observational_available : bool
        For the EIG strategy, whether the model can predict observational
        rows; True for strategies that do not score with the model.

    Raises ValueError (naming the settings) if the set would be empty.
    """
    nodes = resolve_intervenable_nodes(num_variables, intervenable_nodes)
    observational = bool(include_observational and observational_available)
    if not nodes and not observational:
        raise ValueError(
            "the candidate set is empty: intervenable_nodes is empty and the "
            f"observational candidate is not available "
            f"(include_observational_candidate={include_observational}, "
            f"observational prediction available={observational_available})"
        )
    return CandidateSet(nodes, observational)


class ConstrainedEnvironment:
    """An environment that refuses interventions on nodes outside
    `intervenable_nodes` (raises `NonIntervenableNodeError`). The loop always
    wraps the environment in it."""

    def __init__(self, environment: Any, intervenable_nodes: Sequence[int]):
        self.environment = environment
        self.intervenable_nodes = tuple(intervenable_nodes)

    @property
    def num_variables(self) -> int:
        return self.environment.num_variables

    def sample_observational(self, num_rows, generator):
        return self.environment.sample_observational(num_rows, generator)

    def run(self, acquisition: Acquisition, generator):
        node = acquisition.target_node
        if node is not None and node not in self.intervenable_nodes:
            raise NonIntervenableNodeError(
                f"node {node} is not intervenable (intervenable nodes: "
                f"{self.intervenable_nodes})"
            )
        return self.environment.run(acquisition, generator)

    def true_query_distribution(self, treatment, outcome, values):
        return self.environment.true_query_distribution(
            treatment, outcome, values
        )
