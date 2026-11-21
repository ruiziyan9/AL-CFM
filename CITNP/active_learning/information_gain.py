"""Expected information gain (EIG) of candidate experiments about the query,
in the predictive-update form (plan section 3, ABCI Eq. 3.5 and 4.11-4.12):

    EIG(a) = E_psi E_{x ~ p(x | do(a), D)}
             KL( p(y | do(X_T = psi), D + x) || p(y | do(X_T = psi), D) ),

estimated by

    EIG(a) = (1/S) sum_s (1/Q) sum_q KL( p(y | psi_q, D^(s)) || p(y | psi_q, D) )

with S fantasy rows x^(s) from the model's joint prediction and the psi_q the
query grid. The KL between two univariate Gaussian mixtures is computed by
quadrature on a 1-D grid (default) or by Monte Carlo. All densities are on the
raw scale in float64 (the per-task standardisation changes with the context,
so standardised densities of two contexts are not comparable).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from CITNP.active_learning.config import ActiveLearningConfig, CandidateExperiment
from CITNP.active_learning.fantasy import (
    ExperimentContext,
    append_rows_to_context,
    sample_fantasy_rows,
)
from CITNP.active_learning.predictive_model import (
    PredictiveModel,
    QueryMixture,
    predict_query_in_chunks,
)
from CITNP.evaluation.model_api import JointSampleNoise

# The quadrature grid spans the first mixture's components +- this many
# standard deviations.
QUADRATURE_HALF_WIDTH_IN_STANDARD_DEVIATIONS = 8.0
# Largest number of (mixture, grid point, component) entries per quadrature
# chunk, to bound memory.
_QUADRATURE_CHUNK_ENTRIES = 1 << 23


# --------------------------------------------------------------------------- #
# Densities and KL divergences of univariate Gaussian mixtures
# --------------------------------------------------------------------------- #
def mixture_log_density(
    mixture: QueryMixture, points: torch.Tensor
) -> torch.Tensor:
    """log p(points) under univariate mixtures.

    mixture: tensors [..., K]; points [..., G] with the same leading axes.
    Returns [..., G], float64, with `logsumexp` over components."""
    y = points[..., :, None]  # [..., G, 1]
    mean = mixture.mean[..., None, :]
    sd = mixture.standard_deviation[..., None, :]
    log_normal = (
        -0.5 * ((y - mean) / sd) ** 2
        - torch.log(sd)
        - 0.5 * math.log(2 * math.pi)
    )
    return torch.logsumexp(log_normal + mixture.log_weights[..., None, :], dim=-1)


def _flatten(mixture: QueryMixture, lead: torch.Size) -> QueryMixture:
    K = mixture.mean.shape[-1]
    return QueryMixture(
        mixture.mean.expand(*lead, K).reshape(-1, K),
        mixture.standard_deviation.expand(*lead, K).reshape(-1, K),
        mixture.log_weights.expand(*lead, K).reshape(-1, K),
    )


def kl_divergence_quadrature(
    p: QueryMixture, q: QueryMixture, num_points: int = 1024
) -> torch.Tensor:
    """KL(p || q) of univariate Gaussian mixtures by trapezoidal quadrature.

    Grid: `num_points` points from min_k(mu_k - 8 sd_k) to max_k(mu_k + 8 sd_k)
    over the components of p (the integrand p (log p - log q) vanishes
    outside p's support, so only p sets the range), then
        KL = sum_g Delta_g p(y_g) (log p(y_g) - log q(y_g)).

    p, q: tensors [..., K_p] and [..., K_q] whose leading axes broadcast.
    Returns [...] float64 (non-negative up to quadrature error)."""
    lead = torch.broadcast_shapes(p.mean.shape[:-1], q.mean.shape[:-1])
    pf, qf = _flatten(p, lead), _flatten(q, lead)
    M = pf.mean.shape[0]
    K = max(pf.mean.shape[-1], qf.mean.shape[-1])
    chunk = max(1, _QUADRATURE_CHUNK_ENTRIES // (num_points * K))
    width = QUADRATURE_HALF_WIDTH_IN_STANDARD_DEVIATIONS
    unit = torch.linspace(0.0, 1.0, num_points, dtype=torch.float64)
    trapezoid = torch.ones(num_points, dtype=torch.float64)
    trapezoid[0] = trapezoid[-1] = 0.5
    out = []
    for start in range(0, M, chunk):
        pc, qc = pf[start : start + chunk], qf[start : start + chunk]
        low = (pc.mean - width * pc.standard_deviation).min(-1).values
        high = (pc.mean + width * pc.standard_deviation).max(-1).values
        points = low[:, None] + (high - low)[:, None] * unit  # [m, G]
        delta = ((high - low) / (num_points - 1))[:, None] * trapezoid
        log_p = mixture_log_density(pc, points)
        log_q = mixture_log_density(qc, points)
        out.append((delta * log_p.exp() * (log_p - log_q)).sum(-1))
    return torch.cat(out).reshape(lead)


def sample_mixture(
    mixture: QueryMixture, num_samples: int, generator: torch.Generator
) -> torch.Tensor:
    """Samples [..., M] from univariate mixtures with tensors [..., K]."""
    K = mixture.mean.shape[-1]
    lead = mixture.mean.shape[:-1]
    weights = mixture.log_weights.exp().reshape(-1, K)
    component = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    ).reshape(*lead, num_samples)
    mean = mixture.mean.gather(-1, component)
    sd = mixture.standard_deviation.gather(-1, component)
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    return mean + sd * noise


def kl_divergence_monte_carlo(
    p: QueryMixture,
    q: QueryMixture,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """KL(p || q) by Monte Carlo: the mean of log p(y) - log q(y) over
    `num_samples` draws y ~ p. Shapes as in `kl_divergence_quadrature`; can
    be negative by chance."""
    lead = torch.broadcast_shapes(p.mean.shape[:-1], q.mean.shape[:-1])
    pf, qf = _flatten(p, lead), _flatten(q, lead)
    y = sample_mixture(pf, num_samples, generator)  # [M, num_samples]
    log_ratio = mixture_log_density(pf, y) - mixture_log_density(qf, y)
    return log_ratio.mean(-1).reshape(lead)


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


def average_mixture(mixtures: QueryMixture, axis: int) -> QueryMixture:
    """The equal-weight average of the mixtures along leading axis `axis`
    (a mixture of all their components): [..., A, ..., V, K] ->
    [..., ..., V, A * K]."""

    def merge(x):
        x = x.movedim(axis, -2)  # [..., V, A, K]
        return x.reshape(*x.shape[:-2], -1)

    A = mixtures.mean.shape[axis]
    return QueryMixture(
        merge(mixtures.mean),
        merge(mixtures.standard_deviation),
        merge(mixtures.log_weights) - math.log(A),
    )


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
@dataclass
class InformationGainEstimate:
    """EIG estimate of one candidate.

    Attributes:
    ----------
    candidate : CandidateExperiment
        The experiment.

    expected_information_gain : float
        (1/S) sum_s KL_s in nats, KL_s the psi-averaged KL of fantasy s.

    standard_error : float
        std_s(KL_s) / sqrt(S) (unbiased standard deviation).

    per_fantasy_information_gain : torch.Tensor
        [S] float64, the KL_s.

    coherence_gap : float
        psi-averaged KL( p(y | D) || (1/S) sum_s p(y | D^(s)) ), NaN unless
        requested (`compute_coherence_gap`). About 0 for a coherent Bayesian
        predictive (the martingale property); a gap comparable to the EIG
        values means the EIG ranking is not trustworthy.
    """

    candidate: CandidateExperiment
    expected_information_gain: float
    standard_error: float
    per_fantasy_information_gain: torch.Tensor
    coherence_gap: float = float("nan")


def predict_query_on_context(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
) -> QueryMixture:
    """The query p(X_outcome | do(X_treatment = psi), context) on the grid:
    a `QueryMixture` [V, K]."""
    values, flags = context.batch(1)
    with torch.no_grad():
        mixture = predictive_model.predict_query(
            values, flags, treatment, outcome, query_treatment_values
        )
    return mixture[0]


def estimate_expected_information_gain(
    predictive_model: PredictiveModel,
    context: ExperimentContext,
    treatment: int,
    outcome: int,
    query_treatment_values: torch.Tensor,
    candidates: list[CandidateExperiment],
    num_fantasies: int,
    fantasy_noise: Mapping[int | None, JointSampleNoise],
    config: ActiveLearningConfig,
    *,
    query_before: QueryMixture | None = None,
    compute_coherence_gap: bool = False,
    kl_generator: torch.Generator | None = None,
) -> list[InformationGainEstimate]:
    """EIG of every candidate about the query, in one batched pass.

    Per candidate c: S = num_fantasies fantasy rows from
    `sample_fantasy_rows` (noise `fantasy_noise[c.target_node]`, so all
    candidates on one node share their standard noise), fantasy contexts
    D + x^(s) (the rows flagged as intervened on the candidate's node), the
    query predicted on all C x S fantasy contexts in chunks of
    `config.inference_batch_size`, then the psi-averaged KL to the query on
    D (`config.kl_estimator`) per fantasy, its mean and standard error.

    Args:
    ----------
    predictive_model : PredictiveModel
        Gives both the fantasies and the query.

    context : ExperimentContext
        The real context D [N, D].

    treatment, outcome : int
        The query X_outcome | do(X_treatment = psi).

    query_treatment_values : torch.Tensor
        The psi grid [V], raw scale.

    candidates : list[CandidateExperiment]
        C candidates with equal rows_per_experiment.

    num_fantasies : int
        S (>= 2).

    fantasy_noise : Mapping[int | None, JointSampleNoise]
        Standard noise per target node (None: observational), S samples,
        with a row axis for multi-row experiments.

    config : ActiveLearningConfig
        fantasy_mode, kl_estimator, quadrature_points,
        kl_monte_carlo_samples, inference_batch_size.

    query_before : QueryMixture | None
        The query on D [V, K] if already computed.

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
    if num_fantasies < 2:
        raise ValueError("num_fantasies must be >= 2")
    psi = torch.as_tensor(query_treatment_values, dtype=torch.float64)
    if query_before is None:
        query_before = predict_query_on_context(
            predictive_model, context, treatment, outcome, psi
        )
    noises = []
    for c in candidates:
        if c.target_node not in fantasy_noise:
            raise KeyError(f"no fantasy noise for node {c.target_node}")
        noise = fantasy_noise[c.target_node]
        if noise.num_samples != num_fantasies:
            raise ValueError(
                f"the noise of node {c.target_node} has {noise.num_samples} "
                f"samples, not num_fantasies = {num_fantasies}"
            )
        noises.append(noise)
    C, S = len(candidates), num_fantasies
    # [C, S, R, D]
    rows = sample_fantasy_rows(
        predictive_model, context, candidates, noises,
        mode=config.fantasy_mode, batch_size=config.inference_batch_size,
    )  # fmt: skip
    R, D = rows.shape[2], rows.shape[3]
    nodes = torch.tensor(
        [-1 if c.target_node is None else c.target_node for c in candidates]
    ).repeat_interleave(S)
    values, flags = context.batch(C * S)
    values, flags = append_rows_to_context(
        values, flags, rows.reshape(C * S, R, D), nodes
    )
    # [C, S, V, K]
    query_after = predict_query_in_chunks(
        predictive_model, values, flags, treatment, outcome, psi,
        config.inference_batch_size,
    ).reshape(C, S)  # fmt: skip
    kl = kl_divergence(query_after, query_before, config, kl_generator)
    per_fantasy = kl.mean(-1)  # [C, S], averaged over psi
    gain = per_fantasy.mean(-1)
    standard_error = per_fantasy.std(-1, unbiased=True) / math.sqrt(S)
    gaps = torch.full((C,), float("nan"), dtype=torch.float64)
    if compute_coherence_gap:
        average = average_mixture(query_after, axis=1)  # [C, V, S * K]
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
