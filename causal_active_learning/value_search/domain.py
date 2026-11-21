"""The interval of values searched for one node (`ValueDomain`), in a
standardised coordinate u with a raw value v = centre + scale * u."""

from dataclasses import dataclass

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentContext

LENGTHSCALE_BOUNDS_IN_DOMAIN_WIDTHS = (0.05, 2.0)


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


def value_domains(
    context: ExperimentContext,
    nodes: list[int],
    config: ActiveLearningConfig,
    prior_value_range: tuple[float, float] | None = None,
) -> dict[int, ValueDomain]:
    """The searched value interval of every node.

    v = m_i + s_i u with (m_i, s_i) the node's mean and standard deviation
    over the observational rows of `context` (the model's standardisation).
    "observational_range": u in [-c, c], c = value_domain_half_width.
    "prior_range": v in `prior_value_range` (raw scale, supplied by the
    integration from the training prior), mapped to u."""
    mean, std = context.observational_mean_and_standard_deviation()
    domains = {}
    for node in nodes:
        centre, scale = float(mean[node]), float(std[node])
        if config.value_domain == "prior_range":
            if prior_value_range is None:
                raise ValueError(
                    "value_domain 'prior_range' needs prior_value_range"
                )
            low, high = prior_value_range
            lower, upper = (low - centre) / scale, (high - centre) / scale
        else:
            c = config.value_domain_half_width
            lower, upper = -c, c
        domains[node] = ValueDomain(
            node, float(lower), float(upper), centre, scale,
            config.rows_per_experiment,
        )  # fmt: skip
    return domains
