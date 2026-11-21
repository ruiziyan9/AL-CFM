"""The value domain of one node: a standardised coordinate u in [-1, 1]
mapped to a raw value v = centre + half_width * u.

- "observational_range": centre m_i, half width s_i * c, with (m_i, s_i) the
  node's mean and standard deviation over the observational rows of the
  context (the statistics the model's standardisation uses) and
  c = `value_domain_half_width`;
- "training_range": the do-value range the model saw in training (supplied by
  the integration through `TrainingSupport.do_value_range`);
- "fixed_range": `value_domain_fixed_range`, the same for every node.
"""

from dataclasses import dataclass

from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext

LOWER, UPPER = -1.0, 1.0
# The GP lengthscale is bounded to [0.1, 4] in u (domain width 2).
LENGTHSCALE_BOUNDS = (0.1, 4.0)


@dataclass(frozen=True)
class ValueDomain:
    """u in [-1, 1] <-> v = centre + half_width * u for one node.

    Attributes:
    ----------
    target_node : int
        The node.

    centre, half_width : float
        Raw centre and half width of the interval.

    num_rows : int
        R of every acquisition built from this domain.
    """

    target_node: int
    centre: float
    half_width: float
    num_rows: int = 1

    def __post_init__(self):
        if not self.half_width > 0:
            raise ValueError("a value domain needs half_width > 0")

    @property
    def lower(self) -> float:
        """Lower bound of u (-1)."""
        return LOWER

    @property
    def upper(self) -> float:
        """Upper bound of u (1)."""
        return UPPER

    @property
    def width(self) -> float:
        """Width of the u interval (2)."""
        return UPPER - LOWER

    @property
    def raw_range(self) -> tuple[float, float]:
        """(low, high) on the raw scale."""
        return self.centre - self.half_width, self.centre + self.half_width

    def to_value(self, standardised_value: float) -> float:
        """u -> v (raw scale)."""
        return self.centre + self.half_width * float(standardised_value)

    def to_standardised(self, value: float) -> float:
        """v -> u."""
        return (float(value) - self.centre) / self.half_width

    def acquisition(self, standardised_value: float) -> Acquisition:
        """R rows under do(X_node = to_value(u))."""
        return Acquisition.intervention(
            self.target_node, self.to_value(standardised_value), self.num_rows
        )

    def lengthscale_bounds(self) -> tuple[float, float]:
        """GP lengthscale bounds in u."""
        return LENGTHSCALE_BOUNDS


def value_domains(
    context: ExperimentalContext,
    nodes,
    config: ActiveLearningConfig,
    training_value_range: tuple[float, float] | None = None,
) -> dict[int, ValueDomain]:
    """The value domain of every node of `nodes` (see the module docstring)."""
    domains: dict[int, ValueDomain] = {}
    nodes = list(nodes)
    if not nodes:
        return domains
    if config.value_domain == "observational_range":
        mean, std = context.observational_mean_and_standard_deviation()
    for node in nodes:
        if config.value_domain == "observational_range":
            centre = float(mean[node])
            half = float(std[node]) * config.value_domain_half_width
        else:
            if config.value_domain == "fixed_range":
                assert config.value_domain_fixed_range is not None
                low, high = config.value_domain_fixed_range
            else:
                if training_value_range is None:
                    raise ValueError(
                        "value_domain 'training_range' needs the training "
                        "do-value range (TrainingSupport.do_value_range)"
                    )
                low, high = training_value_range
            centre, half = (low + high) / 2, (high - low) / 2
        domains[node] = ValueDomain(
            node, centre, half, config.rows_per_acquisition
        )
    return domains
