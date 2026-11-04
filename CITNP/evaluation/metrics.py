"""How to score a prediction: Monte Carlo KL estimators on a do-value grid,
implied effects (slope of the predictive mean in t), variance ratios.

No model code here; model predictions arrive as mixture parameters from
`model_api.predict_mixture`.
"""

import numpy as np
import torch
from scipy.special import logsumexp

from CITNP.evaluation.oracle import LinearSCM


def obs_cov(scm: LinearSCM) -> np.ndarray:
    """Observational covariance matrix of a linear SCM."""
    A = np.linalg.inv(np.eye(len(scm.s2)) - scm.B.T)
    return A @ np.diag(scm.s2) @ A.T


def naive_slope(S: np.ndarray, T: int, Y: int) -> float:
    """Observational regression slope of X_Y on X_T from covariance S."""
    return float(S[T, Y] / S[T, T])


# --------------------------------------------------------------------------- #
# Densities and samples on a grid of do-values t [n_t], n_y samples per t
# --------------------------------------------------------------------------- #
def mix_logpdf(
    y: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    logw: torch.Tensor,
    chunk: int | None = None,
):
    """log density of y [n_t, n_y] under per-t Gaussian mixtures
    mu, sd, logw [n_t, K]  ->  [n_t, n_y].

    chunk: if set and K > chunk, the logsumexp runs over blocks of `chunk`
    components, combined with logaddexp, so memory is O(n_t n_y chunk)
    instead of O(n_t n_y K). Equal to the unchunked value up to float
    rounding; K <= chunk takes the unchunked path (bit-identical)."""
    K = mu.shape[-1]
    if chunk is None or K <= chunk:
        lp = torch.distributions.Normal(mu[:, None, :], sd[:, None, :]).log_prob(
            y[:, :, None]
        )
        return torch.logsumexp(lp + logw[:, None, :], dim=-1)
    out = None
    for s in range(0, K, chunk):
        sl = slice(s, s + chunk)
        dist = torch.distributions.Normal(mu[:, None, sl], sd[:, None, sl])
        lp = dist.log_prob(y[:, :, None])
        part = torch.logsumexp(lp + logw[:, None, sl], dim=-1)
        out = part if out is None else torch.logaddexp(out, part)
    return out


def sample_truth(
    b: float, v: float, grid: np.ndarray, n_y: int, rng
) -> np.ndarray:
    """Samples [n_t, n_y] of the true do-distribution N(b t, v)."""
    return b * grid[:, None] + np.sqrt(v) * rng.standard_normal((len(grid), n_y))


def truth_logpdf(
    b: float, v: float, g: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """log N(y; b t, v) for y [n_t, n_y], g the grid as a float64 tensor."""
    return torch.distributions.Normal(b * g[:, None], np.sqrt(v)).log_prob(y)


def oracle_weights(log_w: np.ndarray, var: np.ndarray):
    """Normalised (log_w, w) and variances floored at 1e-12 of the oracle's
    predictive components."""
    log_w = log_w - logsumexp(log_w)
    return log_w, np.exp(log_w), np.maximum(var, 1e-12)


def sample_oracle(
    w: np.ndarray,
    slope: np.ndarray,
    var: np.ndarray,
    grid: np.ndarray,
    n_y: int,
    rng,
) -> np.ndarray:
    """Samples [n_t, n_y] of the oracle mixture sum_m w_m N(slope_m t, var_m)."""
    idx = rng.choice(len(w), size=(len(grid), n_y), p=w)
    return slope[idx] * grid[:, None] + np.sqrt(var[idx]) * rng.standard_normal(
        (len(grid), n_y)
    )


def sample_mixture(
    mu: np.ndarray, sd: np.ndarray, logw: np.ndarray, n_y: int, rng
) -> np.ndarray:
    """Samples [n_t, n_y] of per-t Gaussian mixtures with parameters
    mu, sd, logw [n_t, K] (e.g. a model's predictive on a grid of t)."""
    n_t = mu.shape[0]
    w = np.exp(logw - logsumexp(logw, axis=-1, keepdims=True))
    cdf = np.cumsum(w, axis=-1)
    cdf[:, -1] = 1.0
    u = rng.random((n_t, n_y))
    idx = (u[..., None] > cdf[:, None, :]).sum(-1)  # [n_t, n_y]
    rows = np.arange(n_t)[:, None]
    return mu[rows, idx] + sd[rows, idx] * rng.standard_normal((n_t, n_y))


def oracle_mixture_on_grid(log_w, slope, var, g: torch.Tensor):
    """Oracle mixture as (mu, sd, logw) tensors [n_t, M] on g's device."""
    dev = g.device
    o_mu = torch.tensor(slope, device=dev)[None] * g[:, None]
    o_sd = torch.tensor(np.sqrt(var), device=dev)[None].expand_as(o_mu)
    o_lw = torch.tensor(log_w, device=dev)[None].expand_as(o_mu)
    return o_mu, o_sd, o_lw


def kl_mc(lp_p: torch.Tensor, lp_q: torch.Tensor) -> np.ndarray:
    """Monte Carlo KL(p || q) per grid point, from log p and log q at samples
    y ~ p [n_t, n_y] -> [n_t]."""
    return (lp_p - lp_q).mean(1).cpu().numpy()


def kl_mc_mean_with_standard_error(
    lp_p: torch.Tensor, lp_q: torch.Tensor
) -> tuple[float, float]:
    """Mean over grid points of the Monte Carlo KL(p || q) of `kl_mc`, and its
    Monte Carlo standard error, from log p and log q at samples y ~ p
    [n_t, n_y].

    With d = log p - log q, the estimate is mean_t mean_y d and its standard
    error sqrt(sum_t var_y(d) / n_y) / n_t (draws independent across grid
    points; var with ddof 1). It covers the y draws only: if p or q is
    itself a Monte Carlo mixture (the oracle's parameter draws), that error
    is not included. NaN for a single draw per point."""
    d = (lp_p - lp_q).double().cpu().numpy()
    n_t, n_y = d.shape
    if n_y < 2:
        return float(d.mean()), float("nan")
    standard_error = np.sqrt((d.var(axis=1, ddof=1) / n_y).sum()) / n_t
    return float(d.mean()), float(standard_error)


# --------------------------------------------------------------------------- #
# Moments, implied effects, variance ratios
# --------------------------------------------------------------------------- #
def oracle_moments(
    w: np.ndarray, slope: np.ndarray, var: np.ndarray, grid: np.ndarray
):
    """(implied slope mean, slope sd, predictive mean [n_t], predictive var [n_t])."""
    s_bar = float(w @ slope)
    s_sd = float(np.sqrt(max(w @ slope**2 - s_bar**2, 0.0)))
    mean_o = s_bar * grid
    var_o = float(w @ var) + grid**2 * s_sd**2
    return s_bar, s_sd, mean_o, var_o


def mixture_moments(mu: np.ndarray, sd: np.ndarray, logw: np.ndarray):
    """Mean and variance [n_t] of mixtures with parameters [n_t, K]."""
    w = np.exp(logw)
    mean = (w * mu).sum(-1)
    var = (w * (sd**2 + mu**2)).sum(-1) - mean**2
    return mean, var


def implied_slope(grid: np.ndarray, mean: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the least-squares line through mean(t)."""
    slope, icpt = np.polyfit(grid, mean, 1)
    return slope, icpt


def variance_ratio(var_m: np.ndarray, var_o: np.ndarray) -> float:
    """Mean over the grid of var_model(t) / var_oracle(t)."""
    return float(np.mean(var_m / var_o))
