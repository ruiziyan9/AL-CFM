"""Strategies: how the next acquisition is chosen. All share the loop and the
candidate set (`build_candidate_set`).

- "expected_information_gain" (ABCI's U_CR): the bi-level EIG selection of
  selection.py;
- "observational" (OBS): observational rows only;
- "random_fixed_value" (RAND-FIXED): uniform over the intervenable nodes and
  the observational candidate; value `fixed_intervention_value`;
- "random" (RAND): as above; value uniform over `random_value_range` (or the
  node's value domain);
- "treatment": always intervene on T, value uniform over its domain; raises
  if T is not intervenable;
- "exhaustive_true_gain" (evaluation only, an upper reference): the inner
  search per node, then every candidate run `true_gain_draws` times on the
  true system with fresh draws from the reference's own generator; the
  largest mean drop in query KL to the truth wins (per unit cost,
  `cost_adjusted_value`, as the drop may be negative). The chosen candidate
  is then run once more by the loop with the environment generator, which
  the reference never touches, so the real rows are a new draw: the
  reference ranks experiments by their expected true value, not by the
  luck of a draw it has already seen.

The baselines do not score with the model, so they include the observational
candidate whenever `include_observational_candidate` is true (the true system
produces the rows), as ABCI's RAND does.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from causal_active_learning.acquisitions import (
    Acquisition,
    InterventionalQuery,
    NonIntervenableNodeError,
    build_candidate_set,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
    kl_divergence_quadrature,
)
from causal_active_learning.information_gain import predict_query_in_chunks
from causal_active_learning.protocols import (
    CausalPredictiveModel,
    ExperimentEnvironment,
)
from causal_active_learning.selection import (
    AcquisitionScore,
    acquisition_cost,
    choice_is_clear,
    cost_adjusted_value,
    dense_value_diagnostic,
    eig_candidate_set,
    search_intervention_values,
    select_next_acquisition,
)
from causal_active_learning.value_search import ValueEvaluation
from causal_active_learning.value_search.domain import value_domains

if TYPE_CHECKING:
    from causal_active_learning.loop import ActiveLearningGenerators


@dataclass
class StrategyChoice:
    """What a strategy decided in one step.

    Attributes:
    ----------
    chosen : Acquisition
        The acquisition to run.

    scores : list[AcquisitionScore]
        Scores of the candidates (EIG strategies; empty for baselines).

    choice_is_clear : bool | None
        See `choice_is_clear` (None for the baselines).

    predicted_information_gain : float
        Re-estimate of the selected objective for the chosen acquisition
        (NaN for baselines).

    predicted_objective_values : dict[str, float]
        Every objective's re-estimate for the chosen acquisition (EIG
        strategy; empty otherwise).

    dense_value_diagnostic : dict | None
        Dense value diagnostic, when requested.
    """

    chosen: Acquisition
    scores: list[AcquisitionScore] = field(default_factory=list)
    choice_is_clear: bool | None = None
    predicted_information_gain: float = float("nan")
    dense_value_diagnostic: dict[int, list[ValueEvaluation]] | None = None
    predicted_objective_values: dict[str, float] = field(default_factory=dict)


def query_kl(
    truth: UnivariateGaussianMixture,
    model: UnivariateGaussianMixture,
    points: int,
) -> float:
    """psi-averaged KL(truth || model) by quadrature (ABCI's Query KLD)."""
    return float(kl_divergence_quadrature(truth, model, points).mean())


def _uniform(low: float, high: float, generator: torch.Generator) -> float:
    return low + (high - low) * float(
        torch.rand(1, generator=generator, dtype=torch.float64)
    )


def _random_choice(options: list, generator: torch.Generator):
    return options[int(torch.randint(len(options), (1,), generator=generator))]


def choose_acquisition(
    model: CausalPredictiveModel,
    environment: ExperimentEnvironment,
    context: ExperimentalContext,
    query: InterventionalQuery,
    config: ActiveLearningConfig,
    generators: "ActiveLearningGenerators",
    *,
    query_before: UnivariateGaussianMixture,
    truth: UnivariateGaussianMixture | None,
    training_value_range: tuple[float, float] | None = None,
    dense_value_diagnostic_points: int | None = None,
) -> StrategyChoice:
    """The next acquisition of strategy `config.strategy` (see the module
    docstring)."""
    R = config.rows_per_acquisition
    strategy = config.strategy
    if strategy == "expected_information_gain":
        chosen, scores = select_next_acquisition(
            model,
            context,
            query,
            config,
            generators,
            training_value_range=training_value_range,
            query_before=query_before,
            warn_observational_unavailable=False,
        )
        chosen_score = next(s for s in scores if s.acquisition == chosen)
        predicted = chosen_score.expected_information_gain
        diagnostic = None
        if dense_value_diagnostic_points:
            nodes = eig_candidate_set(model, context, config).intervenable_nodes
            domains = value_domains(context, nodes, config, training_value_range)
            diagnostic = dense_value_diagnostic(
                model,
                context,
                query,
                config,
                domains,
                dense_value_diagnostic_points,
                generators.diagnostic,
                query_before=query_before,
            )
        return StrategyChoice(
            chosen,
            scores,
            choice_is_clear(scores),
            predicted,
            diagnostic,
            dict(chosen_score.objective_values),
        )
    if strategy == "observational":
        return StrategyChoice(Acquisition.observational(R))
    candidates = build_candidate_set(
        context.num_variables,
        config.intervenable_nodes,
        config.include_observational_candidate,
        observational_available=True,
    )
    if strategy in ("random", "random_fixed_value"):
        options: list[int | None] = list(candidates.intervenable_nodes)
        if candidates.observational:
            options.append(None)
        node = _random_choice(options, generators.baseline)
        if node is None:
            return StrategyChoice(Acquisition.observational(R))
        if strategy == "random_fixed_value":
            value = config.fixed_intervention_value
        else:
            low_high = config.random_value_range
            if low_high is None:
                domain = value_domains(
                    context, [node], config, training_value_range
                )[node]
                low_high = domain.raw_range
            value = _uniform(*low_high, generators.baseline)
        return StrategyChoice(Acquisition.intervention(node, value, R))
    if strategy == "treatment":
        if query.treatment not in candidates.intervenable_nodes:
            raise NonIntervenableNodeError(
                "strategy 'treatment' needs the treatment "
                f"X{query.treatment + 1} "
                f"to be intervenable (intervenable nodes: "
                f"{candidates.intervenable_nodes})"
            )
        domain = value_domains(
            context, [query.treatment], config, training_value_range
        )[query.treatment]
        value = domain.to_value(_uniform(-1.0, 1.0, generators.baseline))
        return StrategyChoice(Acquisition.intervention(query.treatment, value, R))
    # exhaustive_true_gain
    if truth is None:
        raise ValueError("exhaustive_true_gain needs the true query distribution")
    domains = value_domains(
        context, candidates.intervenable_nodes, config, training_value_range
    )
    inner = search_intervention_values(
        model, context, query, config, generators, domains, query_before
    )
    trials: list[tuple[Acquisition, AcquisitionScore]] = [
        (
            result.acquisition,
            AcquisitionScore(
                result.acquisition,
                result.best_value_surrogate_mean,
                float("nan"),
                float("nan"),
                result,
                acquisition_cost(result.acquisition, config),
            ),
        )
        for result in inner.values()
    ]
    if candidates.observational:
        obs = Acquisition.observational(R)
        trials.append(
            (
                obs,
                AcquisitionScore(
                    obs,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    None,
                    acquisition_cost(obs, config),
                ),
            )
        )
    best_drop, best = -math.inf, None
    kl_before = query_kl(truth, query_before, config.quadrature_points)
    for acquisition, _ in trials:
        drop = cost_adjusted_value(
            expected_true_drop(
                model,
                environment,
                context,
                query,
                acquisition,
                truth,
                kl_before,
                config,
                generators.true_gain,
            ),
            acquisition_cost(acquisition, config),
        )
        if drop > best_drop:
            best_drop, best = drop, acquisition
    assert best is not None
    return StrategyChoice(best, [s for _, s in trials])


def expected_true_drop(
    model: CausalPredictiveModel,
    environment: ExperimentEnvironment,
    context: ExperimentalContext,
    query: InterventionalQuery,
    acquisition: Acquisition,
    truth: UnivariateGaussianMixture,
    kl_before: float,
    config: ActiveLearningConfig,
    generator,
) -> float:
    """Mean drop in query KL to the truth over `config.true_gain_draws`
    fresh runs of `acquisition` on the true system (rows from `generator`,
    never the environment generator of the real step)."""
    node = acquisition.target_node
    contexts = [
        context.append(environment.run(acquisition, generator), node)
        for _ in range(config.true_gain_draws)
    ]
    after = predict_query_in_chunks(
        model,
        contexts,
        query.treatment,
        query.outcome,
        query.treatment_values,
        config.inference_batch_size,
    )
    kl_after = [
        query_kl(truth, after[i], config.quadrature_points)
        for i in range(len(contexts))
    ]
    return kl_before - sum(kl_after) / len(kl_after)
