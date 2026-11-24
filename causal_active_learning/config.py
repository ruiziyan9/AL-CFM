"""Settings of the active learning loop (`ActiveLearningConfig`)."""

from dataclasses import dataclass

STRATEGIES = (
    "expected_information_gain",
    "observational",
    "random_fixed_value",
    "random",
    "treatment",
    "exhaustive_true_gain",
)
FANTASY_NOISE = ("independent", "shared")
KL_ESTIMATORS = ("quadrature", "monte_carlo")
# The objective the inner and outer levels maximise (see
# `information_gain.estimate_expected_information_gain` and the README's
# "Information gain objectives"); all three are always computed and recorded.
INFORMATION_GAIN_OBJECTIVES = (
    "entropy_reduction",
    "kl_divergence",
    "mixture_entropy_reduction",
)
VALUE_SEARCHES = ("bayesian_optimisation", "grid")
VALUE_DOMAINS = ("observational_range", "training_range", "fixed_range")
GAUSSIAN_PROCESS_NOISE = ("auto", "standard_error", "fixed", "fitted")
# num_fantasies below this gives unusable standard errors.
MINIMUM_FANTASIES = 8


def _range(value, name) -> tuple[float, float] | None:
    if value is None:
        return None
    low, high = (float(v) for v in value)
    if not low < high:
        raise ValueError(f"{name} must satisfy low < high, got {value!r}")
    return low, high


@dataclass
class ActiveLearningConfig:
    """Settings of `select_next_acquisition` and `run_active_learning`.

    Attributes:
    ----------
    acquisition_budget : int
        Number of acquisitions (loop steps after step 0).

    rows_per_acquisition : int
        R, rows per acquisition (default 3, ABCI Figure 4's N_t = 3).

    strategy : str
        One of STRATEGIES (see `strategies.choose_acquisition`).

    intervenable_nodes : tuple[int, ...] | None
        The nodes that can be intervened on (0-based); None = all nodes. The
        only setting that decides the interventional candidates, for every
        strategy; leave the treatment or outcome out to exclude them.

    include_observational_candidate : bool
        Also consider an observational acquisition (default). The EIG
        strategy skips it, with one warning, if the model cannot predict
        observational rows.

    observational_cost, intervention_cost : float
        Costs of the two kinds of acquisition; the outer level maximises the
        cost-adjusted value (`selection.cost_adjusted_value`: value / cost
        for a non-negative value, value * cost for a negative one).

    fixed_intervention_value : float
        Value of the "random_fixed_value" strategy (ABCI's RAND-FIXED).

    random_value_range : tuple[float, float] | None
        Raw value range of the "random" strategy (ABCI's RAND); None = the
        node's value domain.

    fantasy_noise : str
        "independent" (default: fresh fantasy noise for every EIG
        evaluation) or "shared" (one draw per node per step, common random
        numbers).

    num_fantasies, final_num_fantasies : int
        Fantasies per EIG evaluation in the value search and at the outer
        re-estimation.

    information_gain_objective : str
        The objective maximised, one of INFORMATION_GAIN_OBJECTIVES, with p0
        the query on D, p_s the query after fantasy s and p_bar their average:
        "entropy_reduction" (default) H(p0) - E_s H(p_s), which may be
        negative for a model that is not coherent; "kl_divergence"
        E_s KL(p_s || p0), the objective before entropy_reduction became the
        default; "mixture_entropy_reduction" H(p_bar) - E_s H(p_s).

    kl_estimator : str
        "quadrature" (default) or "monte_carlo", for the "kl_divergence"
        objective and the coherence gap; entropies and the p_bar terms are
        always computed by quadrature.

    quadrature_points, kl_monte_carlo_samples : int
        Size of the quadrature grid; Monte Carlo samples.

    true_gain_draws : int
        "exhaustive_true_gain" only: fresh runs of each candidate on the true
        system whose mean drop in query KL to the truth ranks the candidates.

    inference_batch_size : int
        Contexts per forward pass.

    value_search : str
        "bayesian_optimisation" (default) or "grid".

    value_domain : str
        "observational_range" (v = m_i + s_i c u, u in [-1, 1]),
        "training_range" (the do-value range of the training prior, from the
        training support) or "fixed_range" (`value_domain_fixed_range`).

    value_domain_half_width : float
        c of the observational range.

    value_domain_fixed_range : tuple[float, float] | None
        Raw (low, high) of "fixed_range" (given iff that domain is used).

    grid_num_values : int
        Evenly spaced grid values over [-1, 1].

    grid_standardised_values : tuple[float, ...] | None
        Explicit grid values in [-1, 1]; take precedence when given.

    bayesian_optimisation_initial_points, bayesian_optimisation_iterations
        Initial values (at -1/2, 0, 1/2 for three) and GP-UCB iterations.

    ucb_exploration_weight : float
        gamma of mu + gamma sigma.

    acquisition_grid_points : int
        Values over [-1, 1] on which the UCB is maximised.

    gaussian_process_restarts : int
        Random starts of the GP hyperparameter fit.

    gaussian_process_noise : str
        "auto" (standard errors plus a bounded nugget under independent
        noise, a bounded fitted nugget under shared noise), or explicitly
        "standard_error", "fixed" (1e-6 only) or "fitted".
    """

    acquisition_budget: int
    rows_per_acquisition: int = 3
    strategy: str = "expected_information_gain"
    intervenable_nodes: tuple[int, ...] | None = None
    include_observational_candidate: bool = True
    observational_cost: float = 1.0
    intervention_cost: float = 1.0
    # baselines
    fixed_intervention_value: float = 0.0
    random_value_range: tuple[float, float] | None = None
    # EIG estimation
    fantasy_noise: str = "independent"
    num_fantasies: int = 64
    final_num_fantasies: int = 256
    information_gain_objective: str = "entropy_reduction"
    kl_estimator: str = "quadrature"
    quadrature_points: int = 1024
    kl_monte_carlo_samples: int = 1024
    inference_batch_size: int = 128
    true_gain_draws: int = 32
    # inner level
    value_search: str = "bayesian_optimisation"
    value_domain: str = "observational_range"
    value_domain_half_width: float = 3.0
    value_domain_fixed_range: tuple[float, float] | None = None
    grid_num_values: int = 9
    grid_standardised_values: tuple[float, ...] | None = None
    bayesian_optimisation_initial_points: int = 3
    bayesian_optimisation_iterations: int = 8
    ucb_exploration_weight: float = 1.0
    acquisition_grid_points: int = 512
    gaussian_process_restarts: int = 5
    gaussian_process_noise: str = "auto"

    def __post_init__(self):
        """Raise ValueError for an unknown string or an invalid number."""
        choices = {
            "strategy": STRATEGIES,
            "fantasy_noise": FANTASY_NOISE,
            "information_gain_objective": INFORMATION_GAIN_OBJECTIVES,
            "kl_estimator": KL_ESTIMATORS,
            "value_search": VALUE_SEARCHES,
            "value_domain": VALUE_DOMAINS,
            "gaussian_process_noise": GAUSSIAN_PROCESS_NOISE,
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, not "
                    f"{getattr(self, name)!r}"
                )
        positive = (
            "acquisition_budget",
            "rows_per_acquisition",
            "num_fantasies",
            "final_num_fantasies",
            "quadrature_points",
            "kl_monte_carlo_samples",
            "inference_batch_size",
            "true_gain_draws",
            "acquisition_grid_points",
            "gaussian_process_restarts",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("observational_cost", "intervention_cost"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be > 0")
        if min(self.num_fantasies, self.final_num_fantasies) < MINIMUM_FANTASIES:
            raise ValueError(
                f"num_fantasies and final_num_fantasies must be >= "
                f"{MINIMUM_FANTASIES} (so standard errors are usable)"
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
        if self.grid_num_values < 2:
            raise ValueError("grid_num_values must be >= 2")
        if self.grid_standardised_values is not None:
            self.grid_standardised_values = tuple(
                float(u) for u in self.grid_standardised_values
            )
            if len(self.grid_standardised_values) < 2:
                raise ValueError("grid_standardised_values needs >= 2 values")
            if any(abs(u) > 1 for u in self.grid_standardised_values):
                raise ValueError("grid_standardised_values must lie in [-1, 1]")
        self.value_domain_fixed_range = _range(
            self.value_domain_fixed_range, "value_domain_fixed_range"
        )
        self.random_value_range = _range(
            self.random_value_range, "random_value_range"
        )
        if (self.value_domain_fixed_range is None) != (
            self.value_domain != "fixed_range"
        ):
            raise ValueError(
                "value_domain_fixed_range must be given if and only if "
                "value_domain is 'fixed_range'"
            )
        if self.intervenable_nodes is not None:
            nodes = tuple(int(n) for n in self.intervenable_nodes)
            if len(set(nodes)) != len(nodes):
                raise ValueError(f"intervenable_nodes has duplicates: {nodes}")
            self.intervenable_nodes = nodes

    def resolved_gaussian_process_noise(self) -> str:
        """The GP noise policy: "auto" resolves to "standard_error" under
        independent fantasy noise and to "fitted" under shared noise."""
        if self.gaussian_process_noise != "auto":
            return self.gaussian_process_noise
        return (
            "standard_error" if self.fantasy_noise == "independent" else "fitted"
        )
