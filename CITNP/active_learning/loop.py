"""The active learning loop (plan sections 1 and 7).

One step, with D_t the real context:
1. predict the query p(Y | do(X_T = psi), D_t) on the psi grid;
2. choose the experiment (`select_next_experiment` for the EIG strategy:
   inner BO per node, then the outer re-evaluation; or a baseline);
3. run it on the environment, append the real rows: D_{t+1} = D_t + rows;
4. predict the query on D_{t+1}; record the realised information gain
   KL(after || before) and the KL to the truth, both averaged over psi.

Randomness: `ActiveLearningGenerators` holds separate generators for the
inner-level fantasy noise, the outer-level fantasy noise, BO, the baselines'
random choices, the dense diagnostic and the environment, so changing the
acquisition settings does not change the real data drawn for a given seed and
sequence of experiments.
"""

import copy
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from CITNP.active_learning.config import (
    ActiveLearningConfig,
    CandidateExperiment,
    candidate_target_nodes,
)
from CITNP.active_learning.design_optimisation import (
    CandidateScore,
    ValueDomain,
    ValueEvaluation,
    ValueOptimisationResult,
    choice_is_clear,
    make_value_optimiser,
    reevaluate_and_select,
)
from CITNP.active_learning.environment import ExperimentEnvironment
from CITNP.active_learning.fantasy import ExperimentContext
from CITNP.active_learning.information_gain import (
    estimate_expected_information_gain,
    kl_divergence_quadrature,
    predict_query_on_context,
)
from CITNP.active_learning.predictive_model import PredictiveModel, QueryMixture
from CITNP.active_learning.training_support import (
    TrainingSupport,
    context_violations,
    worst_case_violations,
)
from CITNP.evaluation.model_api import (
    JointSampleNoise,
    context_standardisation,
    draw_joint_sample_noise,
)

# Central probability mass of the training prior's do-value distribution
# that the "prior_range" value domain covers.
PRIOR_RANGE_TAIL_PROBABILITY = 0.005

# Stream numbers of `ActiveLearningGenerators.from_seed`.
_STREAMS = {
    "inner_fantasy": 1,
    "outer_fantasy": 2,
    "bayesian_optimisation": 3,
    "environment": 4,
    "baseline": 5,
    "diagnostic": 6,
}


def _torch_generator(key: Sequence[int]) -> torch.Generator:
    state = np.random.SeedSequence(list(key)).generate_state(2)
    seed = (int(state[0]) << 32 | int(state[1])) & ((1 << 63) - 1)
    return torch.Generator().manual_seed(seed)


@dataclass
class ActiveLearningGenerators:
    """Separate random streams of one loop run.

    Attributes:
    ----------
    inner_fantasy : torch.Generator
        Fantasy noise of the inner level (one draw per node per step).

    outer_fantasy : torch.Generator
        Fresh fantasy noise of the outer re-evaluation.

    bayesian_optimisation : torch.Generator
        GP hyperparameter restarts.

    environment : np.random.Generator
        The true SCM's noise (`LinearSCM.sample` takes a numpy generator).

    baseline : torch.Generator
        Random choices of the "random" and "treatment" strategies.

    diagnostic : torch.Generator
        Fantasy noise of the dense value diagnostic.
    """

    inner_fantasy: torch.Generator
    outer_fantasy: torch.Generator
    bayesian_optimisation: torch.Generator
    environment: np.random.Generator
    baseline: torch.Generator
    diagnostic: torch.Generator

    @classmethod
    def from_seed(cls, seed: int | Sequence[int]) -> "ActiveLearningGenerators":
        """Independent streams keyed by (seed..., stream number)."""
        key = [int(seed)] if isinstance(seed, int | np.integer) else list(seed)
        return cls(
            inner_fantasy=_torch_generator([*key, _STREAMS["inner_fantasy"]]),
            outer_fantasy=_torch_generator([*key, _STREAMS["outer_fantasy"]]),
            bayesian_optimisation=_torch_generator(
                [*key, _STREAMS["bayesian_optimisation"]]
            ),
            environment=np.random.default_rng([*key, _STREAMS["environment"]]),
            baseline=_torch_generator([*key, _STREAMS["baseline"]]),
            diagnostic=_torch_generator([*key, _STREAMS["diagnostic"]]),
        )


# --------------------------------------------------------------------------- #
# Pieces of a design step
# --------------------------------------------------------------------------- #
def value_domains(
    context: ExperimentContext,
    nodes: list[int],
    config: ActiveLearningConfig,
    prior_do_value_distribution: dict | None = None,
) -> dict[int, ValueDomain]:
    """The searched value interval of every node (plan section 5.2).

    v = m_i + s_i u with (m_i, s_i) the node's mean and standard deviation
    in the model's standardisation of `context` (`context_standardisation`).
    "observational_range": u in [-c, c], c = value_domain_half_width.
    "prior_range": v in the central 1 - 2 * PRIOR_RANGE_TAIL_PROBABILITY
    range of the training prior's do-value distribution of the task (the
    distribution context interventions are drawn from in training, the same
    for every node), mapped to u."""
    mean, std = context_standardisation(context.values, context.intervened)
    low = high = None
    if config.value_domain == "prior_range":
        if prior_do_value_distribution is None:
            raise ValueError(
                "value_domain 'prior_range' needs the do-value distribution "
                "of the training prior (prior_do_value_distribution)"
            )
        from CITNP.datasets.mixed_regime_generator import do_dist_quantile

        q = PRIOR_RANGE_TAIL_PROBABILITY
        low, high = do_dist_quantile(
            prior_do_value_distribution, np.array([q, 1.0 - q])
        )
    domains = {}
    for node in nodes:
        centre, scale = float(mean[node]), float(std[node])
        if low is None or high is None:
            c = config.value_domain_half_width
            lower, upper = -c, c
        else:
            lower, upper = (low - centre) / scale, (high - centre) / scale
        domains[node] = ValueDomain(
            node, float(lower), float(upper), centre, scale,
            config.rows_per_experiment,
        )  # fmt: skip
    return domains


def draw_fantasy_noise(
    nodes: Sequence[int | None],
    num_fantasies: int,
    predictive_model: PredictiveModel,
    rows_per_experiment: int,
    generator: torch.Generator,
) -> dict[int | None, JointSampleNoise]:
    """One independent `JointSampleNoise` per node (None: observational), in
    the given order, with a row axis for multi-row experiments."""
    num_rows = rows_per_experiment if rows_per_experiment > 1 else None
    return {
        node: draw_joint_sample_noise(
            num_fantasies,
            predictive_model.covariance_rank,
            predictive_model.num_variables,
            generator,
            num_rows=num_rows,
        )  # fmt: skip
        for node in nodes
    }


def optimise_intervention_values(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: ActiveLearningGenerators,
    domains: dict[int, ValueDomain],
    query_before: QueryMixture | None = None,
) -> dict[int, ValueOptimisationResult]:
    """Inner level: the most informative value of every node of `domains`
    (config.value_optimiser), with one inner-level noise draw per node
    reused by every evaluation of that node (common random numbers)."""
    noise = draw_fantasy_noise(
        list(domains), config.num_fantasies, predictive_model,
        config.rows_per_experiment, generators.inner_fantasy,
    )  # fmt: skip
    evaluate = partial(
        estimate_expected_information_gain,
        predictive_model, context, treatment, outcome, query_treatment_values,
        num_fantasies=config.num_fantasies, fantasy_noise=noise,
        config=config, query_before=query_before,
        kl_generator=generators.inner_fantasy,
    )  # fmt: skip
    optimiser = make_value_optimiser(config)
    return optimiser.optimise(
        lambda candidates: evaluate(candidates=candidates),
        domains,
        generators.bayesian_optimisation,
    )


def select_next_experiment(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: ActiveLearningGenerators,
    *,
    prior_do_value_distribution: dict | None = None,
    query_before: QueryMixture | None = None,
) -> tuple[CandidateExperiment, list[CandidateScore]]:
    """One design step (the sequential wet-lab mode): the next experiment
    for the query X_outcome | do(X_treatment = psi) on `context`.

    Inner level: per candidate node (`candidate_target_nodes`), the value
    maximising EIG (`optimise_intervention_values`). Outer level: every
    node's recommended experiment (plus an observational one if
    config.include_observational_candidate) re-evaluated with
    config.final_num_fantasies fresh fantasies from the outer-level
    generator, with the coherence gap; the largest re-evaluated EIG wins
    (`reevaluate_and_select`). `choice_is_clear(scores)` says whether the
    runner-up is within two combined standard errors.

    Args:
    ----------
    predictive_model : PredictiveModel
        E.g. `ModelApiPredictiveModel`.

    context : ExperimentContext
        The real data so far.

    treatment, outcome : int
        The query.

    query_treatment_values : torch.Tensor
        The psi grid [V] (raw scale).

    config : ActiveLearningConfig
        Settings.

    generators : ActiveLearningGenerators
        Random streams (inner_fantasy, outer_fantasy, bayesian_optimisation
        are used).

    prior_do_value_distribution : dict | None
        Needed for value_domain "prior_range".

    query_before : QueryMixture | None
        The query on `context` [V, K], if already computed.

    Returns:
    ----------
    (the chosen experiment, one `CandidateScore` per candidate).
    """
    psi = torch.as_tensor(query_treatment_values, dtype=torch.float64)
    nodes = candidate_target_nodes(
        context.num_variables, treatment, outcome, config.exclude_query_nodes
    )
    if not nodes and not config.include_observational_candidate:
        raise ValueError("there is no candidate experiment")
    if query_before is None:
        query_before = predict_query_on_context(
            predictive_model, context, treatment, outcome, psi
        )
    domains = value_domains(context, nodes, config, prior_do_value_distribution)
    inner = optimise_intervention_values(
        predictive_model, context, treatment, outcome, psi, config,
        generators, domains, query_before,
    )  # fmt: skip
    extra = []
    if config.include_observational_candidate:
        extra.append(CandidateExperiment(None, None, config.rows_per_experiment))
    outer_nodes: list[int | None] = [*nodes]
    if extra:
        outer_nodes.append(None)
    outer_noise = draw_fantasy_noise(
        outer_nodes, config.final_num_fantasies, predictive_model,
        config.rows_per_experiment, generators.outer_fantasy,
    )  # fmt: skip
    evaluate_final = partial(
        estimate_expected_information_gain,
        predictive_model, context, treatment, outcome, psi,
        num_fantasies=config.final_num_fantasies, fantasy_noise=outer_noise,
        config=config, query_before=query_before,
        compute_coherence_gap=True, kl_generator=generators.outer_fantasy,
    )  # fmt: skip
    return reevaluate_and_select(
        lambda candidates: evaluate_final(candidates=candidates), inner, extra
    )


def dense_value_diagnostic(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    domains: dict[int, ValueDomain],
    num_values: int,
    generator: torch.Generator,
    query_before: QueryMixture | None = None,
) -> dict[int, list[ValueEvaluation]]:
    """EIG on `num_values` evenly spaced values of every node's domain, with
    config.final_num_fantasies fantasies (one noise per node from
    `generator`), to check that BO finds the optimum of exhaustive search."""
    noise = draw_fantasy_noise(
        list(domains), config.final_num_fantasies, predictive_model,
        config.rows_per_experiment, generator,
    )  # fmt: skip
    plan = [
        (node, float(u))
        for node, d in domains.items()
        for u in np.linspace(d.lower, d.upper, num_values)
    ]
    estimates = estimate_expected_information_gain(
        predictive_model, context, treatment, outcome, query_treatment_values,
        [domains[n].candidate(u) for n, u in plan],
        config.final_num_fantasies, noise, config, query_before=query_before,
        kl_generator=generator,
    )  # fmt: skip
    out: dict[int, list[ValueEvaluation]] = {n: [] for n in domains}
    for (node, u), e in zip(plan, estimates):
        out[node].append(
            ValueEvaluation(
                round=0,
                standardised_value=u,
                value=domains[node].to_value(u),
                expected_information_gain=e.expected_information_gain,
                standard_error=e.standard_error,
            )  # fmt: skip
        )
    return out


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@dataclass
class ActiveLearningStep:
    """The record of one loop step.

    Attributes:
    ----------
    step : int
        1-based; after this step the context holds `step` experiments.

    acquisition : str
        The strategy.

    scores : list[CandidateScore]
        Outer-level scores (EIG strategy; the inner-level surrogate values
        for "exhaustive_true_gain"; empty for the other baselines).

    chosen : CandidateExperiment
        The experiment run.

    choice_is_clear : bool
        See `choice_is_clear` (True for the baselines).

    predicted_information_gain : float
        Re-evaluated EIG of the chosen experiment (NaN for the baselines).

    real_rows : torch.Tensor
        [R, D] rows drawn from the environment.

    query_before, query_after : QueryMixture
        The query [V, K] before and after the rows were added.

    realised_information_gain : float
        psi-averaged KL(after || before).

    kl_to_truth_before, kl_to_truth_after : float
        psi-averaged KL(truth || model) before and after.

    outside_training_support : bool
        The context after the step lies outside the training prior's
        context shapes (`support_violations` says how).

    support_violations : list[str]
        See `context_violations`.

    dense_value_diagnostic : dict[int, list[ValueEvaluation]] | None
        Exhaustive EIG per node on a dense value grid, when requested.
    """

    step: int
    acquisition: str
    scores: list[CandidateScore]
    chosen: CandidateExperiment
    choice_is_clear: bool
    predicted_information_gain: float
    real_rows: torch.Tensor
    query_before: QueryMixture
    query_after: QueryMixture
    realised_information_gain: float
    kl_to_truth_before: float
    kl_to_truth_after: float
    outside_training_support: bool
    support_violations: list[str]
    dense_value_diagnostic: dict[int, list[ValueEvaluation]] | None = None


def _uniform(domain: ValueDomain, generator: torch.Generator) -> float:
    u = float(torch.rand(1, generator=generator, dtype=torch.float64))
    return domain.lower + domain.width * u


def _kl_to_truth(truth: QueryMixture, model: QueryMixture, points: int) -> float:
    return float(kl_divergence_quadrature(truth, model, points).mean())


def _num_target_nodes(acquisition: str, nodes: list[int]) -> int:
    return {"observational": 0, "treatment": 1}.get(acquisition, len(nodes))


def run_active_learning(
    predictive_model: PredictiveModel,
    environment: ExperimentEnvironment,
    initial_context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
    config: ActiveLearningConfig,
    generators: ActiveLearningGenerators,
    *,
    training_support: TrainingSupport | None = None,
    dense_value_diagnostic_steps: Sequence[int] = (),
    dense_value_diagnostic_points: int = 41,
    progress: Callable[[str], None] | None = None,
) -> list[ActiveLearningStep]:
    """Run config.experiment_budget steps of the loop with
    config.acquisition:

    - "expected_information_gain": `select_next_experiment`;
    - "random": a uniformly random candidate node, value uniform over its
      domain (ABCI's RAND);
    - "observational": observational rows only (ABCI's OBS);
    - "treatment": always do(X_treatment), value uniform over its domain;
    - "exhaustive_true_gain" (evaluation only, not a method): the inner level
      for every node, then every node's experiment run on the environment
      with a copy of the environment generator, keeping the one with the
      lowest KL to the truth afterwards. The chosen experiment is then run
      with the real generator, which gives the same rows as its copy.

    Every strategy uses the same environment generator, so strategies run
    with the same seed start from the same random state.

    Args:
    ----------
    predictive_model, environment
        The model and the true SCM.

    initial_context : ExperimentContext
        D_0.

    treatment, outcome : int
        The query.

    query_treatment_values : torch.Tensor
        The psi grid [V] (raw scale).

    config : ActiveLearningConfig
        Settings.

    generators : ActiveLearningGenerators
        Random streams.

    training_support : TrainingSupport | None
        If given, contexts outside it give a warning and the step's
        `outside_training_support` flag (checked for the worst case before
        the loop and for the actual context after every step).

    dense_value_diagnostic_steps, dense_value_diagnostic_points
        Steps (1-based) at which the EIG strategy also runs
        `dense_value_diagnostic` with this many values per node.

    progress : Callable[[str], None] | None
        Called with one line per step (e.g. print).

    Returns:
    ----------
    list[ActiveLearningStep]
    """
    psi = torch.as_tensor(query_treatment_values, dtype=torch.float64)
    D = initial_context.num_variables
    nodes = candidate_target_nodes(
        D, treatment, outcome, config.exclude_query_nodes
    )
    if training_support is not None:
        problems = worst_case_violations(
            initial_context, config.experiment_budget,
            config.rows_per_experiment,
            _num_target_nodes(config.acquisition, nodes), training_support,
        )  # fmt: skip
        for problem in problems:
            warnings.warn(
                f"active learning may leave the training support: {problem}",
                stacklevel=2,
            )
    truth_mean, truth_sd = environment.true_query_distribution(
        treatment, outcome, psi
    )
    truth = QueryMixture.gaussian(truth_mean, truth_sd)
    points = config.quadrature_points
    prior_dist = environment.do_value_distribution
    context = initial_context
    steps = []
    for step in range(1, config.experiment_budget + 1):
        before = predict_query_on_context(
            predictive_model, context, treatment, outcome, psi
        )
        scores: list[CandidateScore] = []
        clear, predicted = True, float("nan")
        diagnostic = None
        acquisition = config.acquisition
        domains: dict[int, ValueDomain] = {}
        if acquisition in ("expected_information_gain", "exhaustive_true_gain"):
            domains = value_domains(context, nodes, config, prior_dist)
        if acquisition == "expected_information_gain":
            chosen, scores = select_next_experiment(
                predictive_model, context, treatment, outcome, psi, config,
                generators, prior_do_value_distribution=prior_dist,
                query_before=before,
            )  # fmt: skip
            clear = choice_is_clear(scores)
            predicted = next(
                s.expected_information_gain
                for s in scores
                if s.candidate == chosen
            )
            if step in dense_value_diagnostic_steps:
                diagnostic = dense_value_diagnostic(
                    predictive_model, context, treatment, outcome, psi,
                    config, domains, dense_value_diagnostic_points,
                    generators.diagnostic, query_before=before,
                )  # fmt: skip
        elif acquisition == "random":
            node = nodes[
                int(
                    torch.randint(len(nodes), (1,), generator=generators.baseline)
                )
            ]
            domain = value_domains(context, [node], config, prior_dist)[node]
            chosen = domain.candidate(_uniform(domain, generators.baseline))
        elif acquisition == "observational":
            chosen = CandidateExperiment(None, None, config.rows_per_experiment)
        elif acquisition == "treatment":
            domain = value_domains(context, [treatment], config, prior_dist)[
                treatment
            ]
            chosen = domain.candidate(_uniform(domain, generators.baseline))
        else:  # exhaustive_true_gain
            inner = optimise_intervention_values(
                predictive_model, context, treatment, outcome, psi, config,
                generators, domains, before,
            )  # fmt: skip
            best_kl = math.inf
            for node, result in inner.items():
                trial = environment.run_experiment(
                    result.candidate, copy.deepcopy(generators.environment)
                )
                after = predict_query_on_context(
                    predictive_model, context.append(trial, node), treatment,
                    outcome, psi,
                )  # fmt: skip
                kl = _kl_to_truth(truth, after, points)
                surrogate_value = result.best_value_posterior_mean
                scores.append(
                    CandidateScore(
                        candidate=result.candidate,
                        expected_information_gain=surrogate_value,
                        standard_error=float("nan"),
                        coherence_gap=float("nan"),
                        inner_result=result,
                    )
                )
                if kl < best_kl:
                    best_kl, chosen = kl, result.candidate
        rows = environment.run_experiment(chosen, generators.environment)
        context_after = context.append(rows, chosen.target_node)
        after = predict_query_on_context(
            predictive_model, context_after, treatment, outcome, psi
        )
        violations = (
            [] if training_support is None
            else context_violations(context_after, training_support)
        )  # fmt: skip
        for problem in violations:
            warnings.warn(
                f"step {step}: outside the training support: {problem}",
                stacklevel=2,
            )
        record = ActiveLearningStep(
            step=step,
            acquisition=acquisition,
            scores=scores,
            chosen=chosen,
            choice_is_clear=clear,
            predicted_information_gain=predicted,
            real_rows=rows,
            query_before=before,
            query_after=after,
            realised_information_gain=float(
                kl_divergence_quadrature(after, before, points).mean()
            ),
            kl_to_truth_before=_kl_to_truth(truth, before, points),
            kl_to_truth_after=_kl_to_truth(truth, after, points),
            outside_training_support=bool(violations),
            support_violations=violations,
            dense_value_diagnostic=diagnostic,
        )
        steps.append(record)
        if progress is not None:
            progress(
                f"[{acquisition}] step {step}: {chosen.describe()}"
                + (
                    f", EIG {predicted:.4f}"
                    + ("" if clear else " (not clear-cut)")
                    if math.isfinite(predicted) else ""
                )
                + f", realised gain {record.realised_information_gain:.4f}, "
                f"KL to truth {record.kl_to_truth_after:.4f}"
            )  # fmt: skip
        context = context_after
    return steps
