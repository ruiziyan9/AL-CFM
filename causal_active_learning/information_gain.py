"""Expected information gain (EIG) of acquisitions about the query, in the
predictive-update form (ABCI Eq. 3.5 and 4.11-4.12), with three objectives.

For one query value psi write p0 = p(y | do(X_T = psi), D), p_s = p(y |
do(X_T = psi), D + x_s) after fantasy s (all R rows of the acquisition, S
fantasies) and p_bar = (1/S) sum_s p_s. Per fantasy, averaged over the psi
values psi_1..psi_Q:

    entropy_reduction            H(p0) - H(p_s)
    kl_divergence                KL(p_s || p0)
    mixture_entropy_reduction    KL(p_s || p_bar)

Each objective's estimate is the mean of its S terms, its standard error
std(terms) / sqrt(S). With I = H(p_bar) - (1/S) sum_s H(p_s), the mutual
information between the query outcome and the fantasy under the model's
fantasy distribution:

    mixture_entropy_reduction = I,
    kl_divergence             = I + KL(p_bar || p0)        (drift),
    entropy_reduction         = I + H(p0) - H(p_bar)       (entropy shift).

All three agree when the model is coherent (p_bar = p0); a PFN does not
guarantee that. `config.information_gain_objective` (default
"entropy_reduction") is the one the value search and the outer level
maximise; all three are computed in every call and recorded.

This is the estimator; the value search (BO or a grid) is the optimiser that
decides where to call it.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from causal_active_learning.acquisitions import Acquisition
from causal_active_learning.config import (
    INFORMATION_GAIN_OBJECTIVES,
    ActiveLearningConfig,
)
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import (
    UnivariateGaussianMixture,
    average_mixture,
    average_mixture_divergences,
    entropy_quadrature,
    kl_divergence_monte_carlo,
    kl_divergence_quadrature,
)
from causal_active_learning.fantasies import fantasy_contexts
from causal_active_learning.protocols import CausalPredictiveModel

# Debugging aid: check the identities between the objectives in every
# estimator call (quadrature KL only), see `check_objective_identities`.
CHECK_OBJECTIVE_IDENTITIES = False
# Largest absolute disagreement (nats) tolerated by the check: quadrature
# error, the objectives' terms being computed on different grids.
OBJECTIVE_IDENTITY_TOLERANCE = 1e-6


def kl_divergence(
    p: UnivariateGaussianMixture,
    q: UnivariateGaussianMixture,
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
    """Information gain estimate of one acquisition.

    Attributes:
    ----------
    acquisition : Acquisition
        The acquisition.

    expected_information_gain : float
        The selected objective (`objective`): (1/S) sum_s G_s in nats, G_s the
        psi-averaged term of fantasy s (module docstring). May be negative
        for "entropy_reduction".

    standard_error : float
        std_s(G_s) / sqrt(S) (unbiased standard deviation).

    per_fantasy_information_gain : torch.Tensor
        [S] float64, the G_s of the selected objective.

    coherence_gap : float
        psi-averaged KL( p(y | D) || (1/S) sum_s p(y | D + x_s) ), NaN unless
        requested. About 0 for a coherent Bayesian predictive; a gap
        comparable to the EIG values means the ranking is not trustworthy.

    objective : str
        The selected objective's name ("" when built by hand, e.g. stubs).

    objective_values, objective_standard_errors : dict[str, float]
        Estimate and standard error of every objective of
        INFORMATION_GAIN_OBJECTIVES. The standard error of
        "mixture_entropy_reduction" is approximate: its terms share p_bar,
        which depends on all S fantasies, so they are not independent.

    drift : float
        psi-averaged KL(p_bar || p0) = kl_divergence -
        mixture_entropy_reduction (0 for a coherent model).

    entropy_shift : float
        psi-averaged H(p0) - H(p_bar) = entropy_reduction -
        mixture_entropy_reduction (0 for a coherent model).
    """

    acquisition: Acquisition
    expected_information_gain: float
    standard_error: float
    per_fantasy_information_gain: torch.Tensor
    coherence_gap: float = float("nan")
    objective: str = ""
    objective_values: dict[str, float] = field(default_factory=dict)
    objective_standard_errors: dict[str, float] = field(default_factory=dict)
    drift: float = float("nan")
    entropy_shift: float = float("nan")


def check_objective_identities(
    objective_values: Mapping[str, torch.Tensor],
    drift: torch.Tensor,
    entropy_shift: torch.Tensor,
    tolerance: float = OBJECTIVE_IDENTITY_TOLERANCE,
) -> None:
    """Raise AssertionError unless, per candidate,
        kl_divergence - mixture_entropy_reduction = drift,
        entropy_reduction - mixture_entropy_reduction = entropy_shift
    hold within `tolerance` (all tensors [C])."""
    mixture = objective_values["mixture_entropy_reduction"]
    errors = {
        "drift": objective_values["kl_divergence"] - mixture - drift,
        "entropy_shift": objective_values["entropy_reduction"]
        - mixture
        - entropy_shift,
    }
    for name, error in errors.items():
        worst = float(error.abs().max()) if error.numel() else 0.0
        if not worst <= tolerance:
            raise AssertionError(
                f"objective identity for {name} off by {worst:.3g} nats "
                f"(tolerance {tolerance:.3g})"
            )


@dataclass
class InformationGainObjectives:
    """Per-fantasy terms of every objective and the two correction terms
    (module docstring), averaged over psi.

    Attributes:
    -----------
    - terms: objective name -> [C, S] float64 (CPU), the terms G_s.
    - drift: [C], KL(p_bar || p0).
    - entropy_shift: [C], H(p0) - H(p_bar).
    """

    terms: dict[str, torch.Tensor]
    drift: torch.Tensor
    entropy_shift: torch.Tensor


def information_gain_objectives(
    query_after: UnivariateGaussianMixture,
    query_before: UnivariateGaussianMixture,
    config: ActiveLearningConfig,
    kl_generator: torch.Generator | None = None,
) -> InformationGainObjectives:
    """The terms of every objective from the query after each fantasy
    [C, S, Q, K] and before [Q, K]: KL(p_s || p0) with `config.kl_estimator`,
    the entropies and the p_bar terms by quadrature with
    `config.quadrature_points` points (no random numbers, so the KL
    estimator's generator is used exactly as by the KL alone)."""
    kl = kl_divergence(query_after, query_before, config, kl_generator)
    points = config.quadrature_points
    entropy_after = entropy_quadrature(query_after, points)  # [C, S, Q]
    entropy_before = entropy_quadrature(query_before, points).to(
        entropy_after.device
    )  # [Q]
    average = average_mixture_divergences(query_after, query_before, points)
    return InformationGainObjectives(
        terms={
            "entropy_reduction": (entropy_before - entropy_after).mean(-1).cpu(),
            "kl_divergence": kl.mean(-1).cpu(),
            "mixture_entropy_reduction": average.kl_to_average.mean(-1).cpu(),
        },
        drift=average.average_kl_to_reference.mean(-1).cpu(),
        entropy_shift=(entropy_before - average.average_entropy).mean(-1).cpu(),
    )


def predict_query_in_chunks(
    model: CausalPredictiveModel,
    contexts: Sequence[ExperimentalContext],
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    batch_size: int,
) -> UnivariateGaussianMixture:
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
                )
            )
    return UnivariateGaussianMixture.concatenate(parts)


def predict_query_on_context(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
) -> UnivariateGaussianMixture:
    """The query on one context: a mixture [Q, K]."""
    with torch.no_grad():
        return model.predict_query(
            [context], treatment, outcome, treatment_values
        )[0]


def estimate_expected_information_gain(
    model: CausalPredictiveModel,
    context: ExperimentalContext,
    treatment: int,
    outcome: int,
    treatment_values: torch.Tensor,
    candidates: Sequence[Acquisition],
    fantasy_noise: Mapping[int | None, Any] | Sequence[Any],
    config: ActiveLearningConfig,
    *,
    query_before: UnivariateGaussianMixture | None = None,
    compute_coherence_gap: bool = False,
    kl_generator: torch.Generator | None = None,
    check_identities: bool | None = None,
) -> list[InformationGainEstimate]:
    """Information gain of every candidate about the query, in one batched
    pass, for every objective (module docstring).

    Per candidate c: S fantasies of all R rows (`model.sample_fantasies` with
    the candidate's noise), the C x S fantasy contexts, the query predicted on
    all of them in chunks of `config.inference_batch_size`, then per fantasy
    and psi: KL(p_s || p0) (`config.kl_estimator`), H(p0) and H(p_s)
    (`entropy_quadrature`), and KL(p_s || p_bar), H(p_bar) and
    KL(p_bar || p0) (`average_mixture_divergences`), all quadratures with
    `config.quadrature_points` points. Each objective's psi-averaged terms
    [C, S] give its mean and standard error; the estimate's headline value is
    `config.information_gain_objective`'s.

    Each term of "mixture_entropy_reduction" is at most log S (p_bar contains
    p_s with weight 1/S), far above EIG values in practice; its standard
    error is approximate (the terms share p_bar).

    Args:
    ----------
    model : CausalPredictiveModel
        Gives both the fantasies and the query.

    context : ExperimentalContext
        The real context.

    treatment, outcome : int
        The query X_outcome | do(X_treatment = psi).

    treatment_values : torch.Tensor
        The psi values [Q], raw scale.

    candidates : Sequence[Acquisition]
        C candidates with equal num_rows.

    fantasy_noise : Mapping[int | None, Any] | Sequence[Any]
        Either one noise per target node (None: observational), shared by
        the candidates on that node, or one noise per candidate, in order.

    config : ActiveLearningConfig
        kl_estimator, quadrature_points, kl_monte_carlo_samples,
        inference_batch_size.

    query_before : UnivariateGaussianMixture | None
        The query on D [Q, K] if already computed.

    compute_coherence_gap : bool
        Also compute `InformationGainEstimate.coherence_gap`.

    kl_generator : torch.Generator | None
        Needed for kl_estimator "monte_carlo".

    check_identities : bool | None
        Check the identities between the objectives
        (`check_objective_identities`; quadrature KL only); None =
        CHECK_OBJECTIVE_IDENTITIES.

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
    objectives = information_gain_objectives(
        query_after, query_before, config, kl_generator
    )
    terms, drift = objectives.terms, objectives.drift
    entropy_shift = objectives.entropy_shift
    means = {name: t.mean(-1) for name, t in terms.items()}
    errors = {
        name: t.std(-1, unbiased=True) / math.sqrt(S) for name, t in terms.items()
    }
    if check_identities is None:
        check_identities = CHECK_OBJECTIVE_IDENTITIES
    if check_identities and config.kl_estimator == "quadrature":
        check_objective_identities(means, drift, entropy_shift)
    objective = config.information_gain_objective
    gaps = torch.full((C,), float("nan"), dtype=torch.float64)
    if compute_coherence_gap:
        mixture = average_mixture(query_after, axis=1)  # [C, Q, S * K]
        gaps = kl_divergence(query_before, mixture, config, kl_generator)
        gaps = gaps.mean(-1).cpu()
    return [
        InformationGainEstimate(
            acquisition=c,
            expected_information_gain=float(means[objective][i]),
            standard_error=float(errors[objective][i]),
            per_fantasy_information_gain=terms[objective][i],
            coherence_gap=float(gaps[i]),
            objective=objective,
            objective_values={
                n: float(means[n][i]) for n in INFORMATION_GAIN_OBJECTIVES
            },
            objective_standard_errors={
                n: float(errors[n][i]) for n in INFORMATION_GAIN_OBJECTIVES
            },
            drift=float(drift[i]),
            entropy_shift=float(entropy_shift[i]),
        )
        for i, c in enumerate(candidates)
    ]
