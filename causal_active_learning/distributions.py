"""Univariate Gaussian mixtures (the query distributions) and the KL
divergence between them.

The KL between two univariate Gaussian mixtures has no closed form; it is
computed by quadrature on a 1-D grid (default) or by Monte Carlo. All
densities are on the raw scale in float64.
"""

import math
from dataclasses import dataclass

import torch

# The quadrature grid spans the first mixture's components +- this many
# standard deviations.
QUADRATURE_HALF_WIDTH_IN_STANDARD_DEVIATIONS = 8.0
# Largest number of (mixture, grid point, component) entries per quadrature
# chunk, to bound memory.
_QUADRATURE_CHUNK_ENTRIES = 1 << 23


@dataclass
class QueryMixture:
    """Univariate Gaussian mixtures on the raw scale, float64:
        p(y) = sum_k exp(log_weights_k) N(y; mean_k, standard_deviation_k^2).

    Attributes:
    -----------
    - mean: [..., K] (for a query: [..., Q, K], Q the treatment values).
    - standard_deviation: [..., K].
    - log_weights: [..., K], normalised over K.
    """

    mean: torch.Tensor
    standard_deviation: torch.Tensor
    log_weights: torch.Tensor

    def __getitem__(self, index) -> "QueryMixture":
        """Index the leading axes of all three tensors."""
        return QueryMixture(
            self.mean[index], self.standard_deviation[index],
            self.log_weights[index],
        )  # fmt: skip

    def reshape(self, *leading: int) -> "QueryMixture":
        """Reshape the axes before (Q, K) to `leading`."""
        tail = self.mean.shape[-2:]
        return QueryMixture(
            self.mean.reshape(*leading, *tail),
            self.standard_deviation.reshape(*leading, *tail),
            self.log_weights.reshape(*leading, *tail),
        )

    @staticmethod
    def concatenate(parts: list["QueryMixture"]) -> "QueryMixture":
        """Concatenate along the first axis."""
        return QueryMixture(
            torch.cat([p.mean for p in parts]),
            torch.cat([p.standard_deviation for p in parts]),
            torch.cat([p.log_weights for p in parts]),
        )

    @staticmethod
    def gaussian(mean, standard_deviation) -> "QueryMixture":
        """One-component mixtures from means and standard deviations [...]."""
        mean = torch.as_tensor(mean, dtype=torch.float64)
        sd = torch.as_tensor(standard_deviation, dtype=torch.float64)
        return QueryMixture(
            mean[..., None], sd[..., None], torch.zeros_like(mean)[..., None]
        )

    def mean_and_variance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixture mean and variance [...]."""
        w = self.log_weights.exp()
        mean = (w * self.mean).sum(-1)
        second = (w * (self.standard_deviation**2 + self.mean**2)).sum(-1)
        return mean, second - mean**2


def mixture_log_density(
    mixture: QueryMixture, points: torch.Tensor
) -> torch.Tensor:
    """log p(points) under univariate mixtures.

    mixture: tensors [..., K]; points [..., G] with the same leading axes.
    Returns [..., G], float64, with `logsumexp` over components, so it stays
    finite far in the tails."""
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
    over the components of p only (the integrand p (log p - log q) is
    negligible outside p's support, and widening the grid for a broad q would
    leave a narrow p with too few points), then
        KL = sum_g Delta_g p(y_g) (log p(y_g) - log q(y_g)),
    with both log densities from `logsumexp` in float64 (finite where q is
    tiny).

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


def average_mixture(mixtures: QueryMixture, axis: int) -> QueryMixture:
    """The equal-weight average of the mixtures along leading axis `axis`
    (a mixture of all their components): [..., A, ..., Q, K] ->
    [..., ..., Q, A * K]."""

    def merge(x):
        x = x.movedim(axis, -2)  # [..., Q, A, K]
        return x.reshape(*x.shape[:-2], -1)

    A = mixtures.mean.shape[axis]
    return QueryMixture(
        merge(mixtures.mean),
        merge(mixtures.standard_deviation),
        merge(mixtures.log_weights) - math.log(A),
    )
