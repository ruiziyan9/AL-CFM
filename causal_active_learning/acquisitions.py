"""Acquisitions (what the loop can do next) and which of them are allowed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Acquisition:
    """One experiment the loop can run.

    Attributes:
    ----------
    target_node : int | None
        0-based node set by do(); None for an observational experiment.

    intervention_value : float | None
        Do-value on the raw data scale; None if and only if observational.

    num_rows : int
        Rows the experiment produces (all with the same target and value).
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
        """True for an experiment without intervention."""
        return self.target_node is None

    def describe(self) -> str:
        """Readable name, e.g. "do(X3 = 1.25)" (1-based variable names)."""
        if self.target_node is None:
            return "observational"
        return f"do(X{self.target_node + 1} = {self.intervention_value:+.4g})"


def candidate_target_nodes(
    num_variables: int, treatment: int, outcome: int, exclude_query_nodes: bool
) -> list[int]:
    """Nodes the loop may intervene on: every node except the outcome, and
    except the treatment too when `exclude_query_nodes` (default). The
    outcome is never a candidate: setting it removes its response to the
    treatment, so the experiment says nothing about the query."""
    excluded = {outcome, treatment} if exclude_query_nodes else {outcome}
    return [j for j in range(num_variables) if j not in excluded]
