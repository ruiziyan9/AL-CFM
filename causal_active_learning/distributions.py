"""Univariate Gaussian mixtures (the query distributions), their entropy and
the KL divergence between them.

Neither the KL between two univariate Gaussian mixtures nor a mixture's
entropy has a closed form; both are computed by trapezoidal quadrature on a
1-D grid (`quadrature_grid`), the KL also by Monte Carlo. All densities are
on the raw scale in float64.
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
class UnivariateGaussianMixture:
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

    def __getitem__(self, index) -> "UnivariateGaussianMixture":
        """Index the leading axes of all three tensors."""
        return UnivariateGaussianMixture(
            self.mean[index],
            self.standard_deviation[index],
            self.log_weights[index],
        )

    def reshape(self, *leading: int) -> "UnivariateGaussianMixture":
        """Reshape the axes before (Q, K) to `leading`."""
        tail = self.mean.shape[-2:]
        return UnivariateGaussianMixture(
            self.mean.reshape(*leading, *tail),
            self.standard_deviation.reshape(*leading, *tail),
            self.log_weights.reshape(*leading, *tail),
        )

    @staticmethod
    def concatenate(
        parts: list["UnivariateGaussianMixture"],
    ) -> "UnivariateGaussianMixture":
        """Concatenate along the first axis."""
        return UnivariateGaussianMixture(
            torch.cat([p.mean for p in parts]),
            torch.cat([p.standard_deviation for p in parts]),
            torch.cat([p.log_weights for p in parts]),
        )

    @staticmethod
    def gaussian(mean, standard_deviation) -> "UnivariateGaussianMixture":
        """One-component mixtures from means and standard deviations [...]."""
        mean = torch.as_tensor(mean, dtype=torch.float64)
        sd = torch.as_tensor(standard_deviation, dtype=torch.float64)
        return UnivariateGaussianMixture(
            mean[..., None], sd[..., None], torch.zeros_like(mean)[..., None]
        )

    def to(self, device) -> "UnivariateGaussianMixture":
        """The same mixtures on `device`."""
        return UnivariateGaussianMixture(
            self.mean.to(device),
            self.standard_deviation.to(device),
            self.log_weights.to(device),
        )

    def mean_and_variance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixture mean and variance [...]."""
        w = self.log_weights.exp()
        mean = (w * self.mean).sum(-1)
        second = (w * (self.standard_deviation**2 + self.mean**2)).sum(-1)
        return mean, second - mean**2


def mixture_log_density(
    mixture: UnivariateGaussianMixture, points: torch.Tensor
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


def _flatten(
    mixture: UnivariateGaussianMixture, lead: torch.Size
) -> UnivariateGaussianMixture:
    K = mixture.mean.shape[-1]
    return UnivariateGaussianMixture(
        mixture.mean.expand(*lead, K).reshape(-1, K),
        mixture.standard_deviation.expand(*lead, K).reshape(-1, K),
        mixture.log_weights.expand(*lead, K).reshape(-1, K),
    )


def quadrature_grid(
    mixture: UnivariateGaussianMixture, num_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trapezoidal quadrature grid of univariate mixtures: `num_points`
    evenly spaced points from min_k(mu_k - 8 sd_k) to max_k(mu_k + 8 sd_k)
    over the mixture's own components, and the trapezoidal weights.

    mixture: tensors [..., K]. Returns (points, weights), both [..., G]
    float64 on the mixture's device, with sum_g weights_g f(points_g) the
    integral of f over the grid's range."""
    width = QUADRATURE_HALF_WIDTH_IN_STANDARD_DEVIATIONS
    device = mixture.mean.device
    unit = torch.linspace(
        0.0, 1.0, num_points, dtype=torch.float64, device=device
    )
    trapezoid = torch.ones(num_points, dtype=torch.float64, device=device)
    trapezoid[0] = trapezoid[-1] = 0.5
    low = (mixture.mean - width * mixture.standard_deviation).min(-1).values
    high = (mixture.mean + width * mixture.standard_deviation).max(-1).values
    points = low[..., None] + (high - low)[..., None] * unit  # [..., G]
    weights = ((high - low) / (num_points - 1))[..., None] * trapezoid
    return points, weights


def kl_divergence_quadrature(
    p: UnivariateGaussianMixture,
    q: UnivariateGaussianMixture,
    num_points: int = 1024,
) -> torch.Tensor:
    """KL(p || q) of univariate Gaussian mixtures by trapezoidal quadrature.

    Grid: `quadrature_grid` of p only (the integrand p (log p - log q) is
    negligible outside p's support, and widening the grid for a broad q would
    leave a narrow p with too few points), then
        KL = sum_g Delta_g p(y_g) (log p(y_g) - log q(y_g)),
    with both log densities from `logsumexp` in float64 (finite where q is
    tiny).

    p, q: tensors [..., K_p] and [..., K_q] whose leading axes broadcast, on
    one device (q is moved to p's). Returns [...] float64 on p's device
    (non-negative up to quadrature error)."""
    q = q.to(p.mean.device)
    lead = torch.broadcast_shapes(p.mean.shape[:-1], q.mean.shape[:-1])
    pf, qf = _flatten(p, lead), _flatten(q, lead)
    M = pf.mean.shape[0]
    K = max(pf.mean.shape[-1], qf.mean.shape[-1])
    chunk = max(1, _QUADRATURE_CHUNK_ENTRIES // (num_points * K))
    out = []
    for start in range(0, M, chunk):
        pc, qc = pf[start : start + chunk], qf[start : start + chunk]
        points, delta = quadrature_grid(pc, num_points)  # [m, G]
        log_p = mixture_log_density(pc, points)
        log_q = mixture_log_density(qc, points)
        out.append((delta * log_p.exp() * (log_p - log_q)).sum(-1))
    return torch.cat(out).reshape(lead)


def entropy_quadrature(
    mixture: UnivariateGaussianMixture, num_points: int = 1024
) -> torch.Tensor:
    """Differential entropy H(p) = -integral p log p of univariate Gaussian
    mixtures by trapezoidal quadrature on `quadrature_grid` of the mixture,
    with the log density from `logsumexp` in float64.

    mixture: tensors [..., K]. Returns [...] float64 on the mixture's device
    (in nats; may be negative for narrow distributions)."""
    lead = mixture.mean.shape[:-1]
    flat = _flatten(mixture, lead)
    M, K = flat.mean.shape
    chunk = max(1, _QUADRATURE_CHUNK_ENTRIES // (num_points * K))
    out = []
    for start in range(0, M, chunk):
        part = flat[start : start + chunk]
        points, delta = quadrature_grid(part, num_points)  # [m, G]
        log_p = mixture_log_density(part, points)
        out.append(-(delta * log_p.exp() * log_p).sum(-1))
    return torch.cat(out).reshape(lead)


@dataclass
class AverageMixtureDivergences:
    """Divergences involving the equal-weight average p_bar = (1/S) sum_s p_s
    of S mixtures p_s, against a reference r (see
    `average_mixture_divergences`).

    Attributes:
    -----------
    - kl_to_average: [C, S, Q], KL(p_s || p_bar).
    - average_entropy: [C, Q], H(p_bar).
    - average_kl_to_reference: [C, Q], KL(p_bar || r).
    """

    kl_to_average: torch.Tensor
    average_entropy: torch.Tensor
    average_kl_to_reference: torch.Tensor


def average_mixture_divergences(
    mixtures: UnivariateGaussianMixture,
    reference: UnivariateGaussianMixture,
    num_points: int = 1024,
) -> AverageMixtureDivergences:
    """KL(p_s || p_bar), H(p_bar) and KL(p_bar || r) for the average p_bar of
    the mixtures along axis 1, all by trapezoidal quadrature on one grid per
    (c, q): `quadrature_grid` of p_bar, which covers every p_s.

    One shared grid keeps the cost of the S x (S K)-component evaluations
    linear in S: p_bar is evaluated once per (c, q) instead of once per
    fantasy. It spans every p_s's own grid, so each p_s is resolved at least
    as finely as the spread of the S components allows (`num_points` over
    their joint range; the trapezoidal rule on Gaussians is accurate far
    below one point per standard deviation).

    mixtures: tensors [C, S, Q, K]; reference: tensors whose leading axes
    broadcast to [C, Q] (e.g. [Q, K']), moved to the mixtures' device.
    Returns float64 on the mixtures' device."""
    reference = reference.to(mixtures.mean.device)
    C, S, Q, K = mixtures.mean.shape
    average = average_mixture(mixtures, axis=1)  # [C, Q, S K]
    flat_average = _flatten(average, torch.Size([C, Q]))  # [C Q, S K]
    flat_reference = _flatten(reference, torch.Size([C, Q]))

    def by_query(x):  # [C, S, Q, K] -> [C Q, S, K]
        return x.movedim(1, 2).reshape(C * Q, S, K)

    components = UnivariateGaussianMixture(
        by_query(mixtures.mean),
        by_query(mixtures.standard_deviation),
        by_query(mixtures.log_weights),
    )
    M = C * Q
    chunk = max(1, _QUADRATURE_CHUNK_ENTRIES // (2 * num_points * S * K))
    kl_to_average, entropy, kl_to_reference = [], [], []
    for start in range(0, M, chunk):
        part = slice(start, start + chunk)
        bar, comp, ref = (
            flat_average[part],
            components[part],
            flat_reference[part],
        )
        points, delta = quadrature_grid(bar, num_points)  # [m, G]
        log_bar = mixture_log_density(bar, points)
        log_ref = mixture_log_density(ref, points)
        log_comp = mixture_log_density(
            comp, points[:, None, :].expand(-1, S, -1)
        )  # [m, S, G]
        density_bar = log_bar.exp()
        entropy.append(-(delta * density_bar * log_bar).sum(-1))
        kl_to_reference.append(
            (delta * density_bar * (log_bar - log_ref)).sum(-1)
        )
        kl_to_average.append(
            (
                delta[:, None] * log_comp.exp() * (log_comp - log_bar[:, None, :])
            ).sum(-1)
        )
    return AverageMixtureDivergences(
        kl_to_average=torch.cat(kl_to_average).reshape(C, Q, S).movedim(2, 1),
        average_entropy=torch.cat(entropy).reshape(C, Q),
        average_kl_to_reference=torch.cat(kl_to_reference).reshape(C, Q),
    )


def sample_mixture(
    mixture: UnivariateGaussianMixture,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Samples [..., M] from univariate mixtures with tensors [..., K]."""
    K = mixture.mean.shape[-1]
    lead = mixture.mean.shape[:-1]
    weights = mixture.log_weights.exp().reshape(-1, K).cpu()
    component = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    ).reshape(*lead, num_samples)
    component = component.to(mixture.mean.device)
    mean = mixture.mean.gather(-1, component)
    sd = mixture.standard_deviation.gather(-1, component)
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    return mean + sd * noise.to(mean.device)


def kl_divergence_monte_carlo(
    p: UnivariateGaussianMixture,
    q: UnivariateGaussianMixture,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """KL(p || q) by Monte Carlo: the mean of log p(y) - log q(y) over
    `num_samples` draws y ~ p. Shapes as in `kl_divergence_quadrature`; can
    be negative by chance."""
    q = q.to(p.mean.device)
    lead = torch.broadcast_shapes(p.mean.shape[:-1], q.mean.shape[:-1])
    pf, qf = _flatten(p, lead), _flatten(q, lead)
    y = sample_mixture(pf, num_samples, generator)  # [M, num_samples]
    log_ratio = mixture_log_density(pf, y) - mixture_log_density(qf, y)
    return log_ratio.mean(-1).reshape(lead)


def average_mixture(
    mixtures: UnivariateGaussianMixture, axis: int
) -> UnivariateGaussianMixture:
    """The equal-weight average of the mixtures along leading axis `axis`
    (a mixture of all their components): [..., A, ..., Q, K] ->
    [..., ..., Q, A * K]."""

    def merge(x):
        x = x.movedim(axis, -2)  # [..., Q, A, K]
        return x.reshape(*x.shape[:-2], -1)

    A = mixtures.mean.shape[axis]
    return UnivariateGaussianMixture(
        merge(mixtures.mean),
        merge(mixtures.standard_deviation),
        merge(mixtures.log_weights) - math.log(A),
    )
