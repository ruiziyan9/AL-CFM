"""Expected information gain (EIG) of acquisitions about the query, in the
predictive-update form (ABCI Eq. 3.5 and 4.11-4.12):

    EIG(a) = E_psi E_{x ~ p(x | a, D)}
             KL( p(y | do(X_T = psi), D + x) || p(y | do(X_T = psi), D) ),

estimated with S fantasies x_s (all R rows of the acquisition) and the query
values psi_1..psi_Q:

    EIG_hat(a) = (1/S) sum_s KL_s,
    KL_s = (1/Q) sum_q KL( p(y | psi_q, D + x_s) || p(y | psi_q, D) ),
    standard error = std(KL_1..KL_S) / sqrt(S).

This is the estimator; the value search (BO or a grid) is the optimiser that
decides where to call it.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentContext
from causal_active_learning.distributions import (
    QueryMixture,
    average_mixture,
    kl_divergence_monte_carlo,
    kl_divergence_quadrature,
)
from causal_active_learning.fantasies import fantasy_contexts
from causal_active_learning.protocols import PredictiveModel


def kl_divergence(
    p: QueryMixture,
    q: QueryMixture,
    config: ActiveLearningConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """KL(p || q) with the estimator chosen by `config.kl_estimator`
    ("monte_carlo" needs `generator`)."""
    if config.kl_estimator == "quadrature":
        return kl_divergence_quadrature(p, q, config.quadrature_points)
    if generator is None:
        raise ValueError("the Monte Carlo KL needs a generator")
    return kl_divergence_monte_carlo(
        p, q, config.kl_monte_carlo_samples, generator
    )


@dataclass
class InformationGainEstimate:
    """EIG estimate of one acquisition.

    Attributes:
    ----------
    candidate : CandidateExperiment
        The acquisition.

    expected_information_gain : float
        (1/S) sum_s KL_s in nats, KL_s the psi-averaged KL of fantasy s.

    standard_error : float
        std_s(KL_s) / sqrt(S) (unbiased standard deviation).

    per_fantasy_information_gain : torch.Tensor
        [S] float64, the KL_s.

    coherence_gap : float
        psi-averaged KL( p(y | D) || (1/S) sum_s p(y | D + x_s) ), NaN unless
        requested. About 0 for a coherent Bayesian predictive; a gap
        comparable to the EIG values means the ranking is not trustworthy.
    """

    candidate: CandidateExperiment
    expected_information_gain: float
    standard_error: float
    per_fantasy_information_gain: torch.Tensor
    coherence_gap: float = float("nan")


def predict_query_in_chunks(
    model: PredictiveModel,
    contexts: Sequence[ExperimentContext],
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    batch_size: int,
) -> QueryMixture:
    """`predict_query` on a list of equal-size contexts in chunks of
    `batch_size`, under torch.no_grad(): a mixture [B, Q, K]."""
    parts = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            parts.append(
                model.predict_query(
                    contexts[start : start + batch_size],
                    treatment,
                    outcome,
                    treatment_values,
                )  # fmt: skip
            )
    return QueryMixture.concatenate(parts)


def predict_query_on_context(
    model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
) -> QueryMixture:
    """The query on one context: a mixture [Q, K]."""
    with torch.no_grad():
        return model.predict_query(
            [context], treatment, outcome, treatment_values
        )[0]


def estimate_expected_information_gain(
    model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    candidates: Sequence[CandidateExperiment],
    fantasy_noise: Mapping[int | None, Any] | Sequence[Any],
    config: ActiveLearningConfig,
    *,
    query_before: QueryMixture | None = None,
    compute_coherence_gap: bool = False,
    kl_generator: torch.Generator | None = None,
) -> list[InformationGainEstimate]:
    """EIG of every candidate about the query, in one batched pass.

    Per candidate c: S fantasies of all R rows (`model.sample_fantasies` with
    the candidate's noise), the C x S fantasy contexts, the query predicted on
    all of them in chunks of `config.inference_batch_size`, then the
    psi-averaged KL to the query on D (`config.kl_estimator`) per fantasy,
    its mean and standard error.

    Args:
    ----------
    model : PredictiveModel
        Gives both the fantasies and the query.

    context : ExperimentContext
        The real context.

    treatment, outcome : int
        The query X_outcome | do(X_treatment = psi).

    treatment_values : torch.Tensor
        The psi values [Q], raw scale.

    candidates : Sequence[CandidateExperiment]
        C candidates with equal rows_per_experiment.

    fantasy_noise : Mapping[int | None, Any] | Sequence[Any]
        Either one noise per target node (None: observational), shared by
        the candidates on that node, or one noise per candidate, in order.

    config : ActiveLearningConfig
        kl_estimator, quadrature_points, kl_monte_carlo_samples,
        inference_batch_size.

    query_before : QueryMixture | None
        The query on D [Q, K] if already computed.

    compute_coherence_gap : bool
        Also compute `InformationGainEstimate.coherence_gap`.

    kl_generator : torch.Generator | None
        Needed for kl_estimator "monte_carlo".

    Returns:
    ----------
    list[InformationGainEstimate]
        One per candidate, in order.
    """
    if not candidates:
        return []
    candidates = list(candidates)
    psi = torch.as_tensor(treatment_values, dtype=torch.float64)
    if query_before is None:
        query_before = predict_query_on_context(
            model, context, treatment, outcome, psi
        )
    if isinstance(fantasy_noise, Mapping):
        noises = []
        for c in candidates:
            if c.target_node not in fantasy_noise:
                raise KeyError(f"no fantasy noise for node {c.target_node}")
            noises.append(fantasy_noise[c.target_node])
    else:
        noises = list(fantasy_noise)
        if len(noises) != len(candidates):
            raise ValueError("give one fantasy noise per candidate")
    with torch.no_grad():
        fantasies = model.sample_fantasies(
            [context] * len(candidates), candidates, noises
        )  # [C, S, R, D]
    C, S = fantasies.shape[0], fantasies.shape[1]
    if S < 2:
        raise ValueError("at least two fantasies are needed")
    contexts = fantasy_contexts(context, candidates, fantasies)
    query_after = predict_query_in_chunks(
        model, contexts, treatment, outcome, psi, config.inference_batch_size
    ).reshape(C, S)  # [C, S, Q, K]
    kl = kl_divergence(query_after, query_before, config, kl_generator)
    per_fantasy = kl.mean(-1)  # [C, S], averaged over psi
    gain = per_fantasy.mean(-1)
    standard_error = per_fantasy.std(-1, unbiased=True) / math.sqrt(S)
    gaps = torch.full((C,), float("nan"), dtype=torch.float64)
    if compute_coherence_gap:
        average = average_mixture(query_after, axis=1)  # [C, Q, S * K]
        gaps = kl_divergence(query_before, average, config, kl_generator)
        gaps = gaps.mean(-1)
    return [
        InformationGainEstimate(
            candidate=c,
            expected_information_gain=float(gain[i]),
            standard_error=float(standard_error[i]),
            per_fantasy_information_gain=per_fantasy[i],
            coherence_gap=float(gaps[i]),
        )
        for i, c in enumerate(candidates)
    ]
