"""Settings of the active learning loop (`ActiveLearningConfig`)."""

from dataclasses import dataclass

STRATEGIES = (
    "expected_information_gain",
    "random",
    "observational",
    "treatment",
    "exhaustive_true_gain",
)
KL_ESTIMATORS = ("quadrature", "monte_carlo")
VALUE_SEARCHES = ("bayesian_optimisation", "grid")
VALUE_DOMAINS = ("observational_range", "prior_range")


@dataclass
class ActiveLearningConfig:
    """Settings of `select_next_acquisition` and `run_active_learning`.

    Attributes:
    ----------
    acquisition_budget : int
        Number of experiments (loop steps).

    rows_per_acquisition : int
        Rows produced by one experiment.

    acquisition : str
        One of STRATEGIES (see `run_active_learning`).

    exclude_query_nodes : bool
        True: neither the treatment nor the outcome is a candidate node.
        False: the treatment is a candidate too. The outcome never is.

    include_observational_candidate : bool
        Also score an observational experiment at the outer level (needs a
        predictive model with an observational prediction).

    num_fantasies, final_num_fantasies : int
        Fantasies per EIG evaluation at the inner level (BO) and at the
        outer re-evaluation.

    kl_estimator : str
        "quadrature" (default) or "monte_carlo".

    quadrature_points : int
        Grid points of the quadrature KL.

    kl_monte_carlo_samples : int
        Samples of the Monte Carlo KL.

    inference_batch_size : int
        Contexts per forward pass.

    value_search : str
        Inner level: "bayesian_optimisation" or "grid".

    value_domain : str
        "observational_range" ([-half width, half width] in the node's
        standardised coordinate) or "prior_range" (the central range of the
        training prior's do-value distribution).

    value_domain_half_width : float
        Half width c of the observational range.

    grid_standardised_values : tuple[float, ...]
        Values of the grid optimiser, in the standardised coordinate.

    bayesian_optimisation_initial_points, bayesian_optimisation_iterations
        Initial design size and GP-UCB iterations per node.

    ucb_exploration_weight : float
        gamma of the upper confidence bound mu + gamma sigma.

    acquisition_grid_points : int
        Grid on which the upper confidence bound is maximised.

    gaussian_process_restarts : int
        Random starts of the GP hyperparameter fit.
    """

    acquisition_budget: int
    rows_per_acquisition: int = 1
    strategy: str = "expected_information_gain"
    exclude_query_nodes: bool = True
    include_observational_candidate: bool = False
    # EIG estimation
    num_fantasies: int = 64
    final_num_fantasies: int = 256
    kl_estimator: str = "quadrature"
    quadrature_points: int = 1024
    kl_monte_carlo_samples: int = 1024
    inference_batch_size: int = 128
    # inner level
    value_search: str = "bayesian_optimisation"
    value_domain: str = "observational_range"
    value_domain_half_width: float = 3.0
    grid_standardised_values: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    bayesian_optimisation_initial_points: int = 3
    bayesian_optimisation_iterations: int = 8
    ucb_exploration_weight: float = 1.0
    acquisition_grid_points: int = 512
    gaussian_process_restarts: int = 5

    def __post_init__(self):
        """Raise ValueError for an unknown string or an invalid number."""
        choices = {
            "strategy": STRATEGIES,
            "kl_estimator": KL_ESTIMATORS,
            "value_search": VALUE_SEARCHES,
            "value_domain": VALUE_DOMAINS,
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, not "
                    f"{getattr(self, name)!r}"
                )
        positive = (
            "acquisition_budget", "rows_per_acquisition", "num_fantasies",
            "final_num_fantasies", "quadrature_points",
            "kl_monte_carlo_samples", "inference_batch_size",
            "acquisition_grid_points", "gaussian_process_restarts",
        )  # fmt: skip
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.num_fantasies < 2 or self.final_num_fantasies < 2:
            raise ValueError(
                "num_fantasies and final_num_fantasies must be >= 2 (the "
                "standard error needs two fantasies)"
            )
        if self.quadrature_points < 2:
            raise ValueError("quadrature_points must be >= 2")
        if self.bayesian_optimisation_initial_points < 2:
            raise ValueError("bayesian_optimisation_initial_points must be >= 2")
        if self.bayesian_optimisation_iterations < 0:
            raise ValueError("bayesian_optimisation_iterations must be >= 0")
        if not self.value_domain_half_width > 0:
            raise ValueError("value_domain_half_width must be > 0")
        if self.ucb_exploration_weight < 0:
            raise ValueError("ucb_exploration_weight must be >= 0")
        self.grid_standardised_values = tuple(
            float(u) for u in self.grid_standardised_values
        )
        if not self.grid_standardised_values:
            raise ValueError("grid_standardised_values must not be empty")
