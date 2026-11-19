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


# --------------------------------------------------------------------------- #
# Joint interventional distributions (joint models), over the non-intervened
# variables only. The truth is X | do(X_T = t) ~ N(slopes t, covariance)
# (`LinearSCM.do_joint_dist`); the model's joint log densities come from
# `model_api.joint_log_density` and its marginals from
# `model_api.joint_marginal_mixture`.
# --------------------------------------------------------------------------- #
def _non_intervened(D: int, T: int) -> np.ndarray:
    return np.array([j for j in range(D) if j != T], dtype=int)


def sample_truth_joint(
    slopes: np.ndarray,
    covariance: np.ndarray,
    T: int,
    grid: np.ndarray,
    n_y: int,
    rng,
) -> np.ndarray:
    """Full rows [n_t, n_y, D] of the true interventional joint at each
    do-value t of `grid`: non-intervened variables ~ N(slopes t, covariance)
    (their block), X_T = t exactly."""
    D = len(slopes)
    keep = _non_intervened(D, T)
    chol = np.linalg.cholesky(covariance[np.ix_(keep, keep)])
    rows = slopes[None, None, :] * grid[:, None, None]
    rows = np.repeat(rows, n_y, axis=1)  # [n_t, n_y, D]
    noise = rng.standard_normal((len(grid), n_y, len(keep))) @ chol.T
    rows[..., keep] += noise
    rows[..., T] = grid[:, None]
    return rows


def truth_joint_logpdf(
    slopes: np.ndarray,
    covariance: np.ndarray,
    T: int,
    g: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """log N(rows of the non-intervened variables; slopes t, covariance) for
    rows [n_t, n_y, D] at do-values g [n_t] (float64 tensors) -> [n_t, n_y]."""
    keep = _non_intervened(len(slopes), T)
    dev = rows.device
    mean = torch.tensor(slopes[keep], device=dev)[None] * g[:, None]  # [n_t, d]
    cov = torch.tensor(covariance[np.ix_(keep, keep)], device=dev)
    dist = torch.distributions.MultivariateNormal(mean[:, None], cov)
    return dist.log_prob(rows[..., torch.as_tensor(keep, device=dev)])


def kl_truth_joint_model(
    lp_truth: torch.Tensor, lp_model_joint: torch.Tensor
) -> np.ndarray:
    """Monte Carlo KL(truth || model) of the joint over the non-intervened
    variables, per do-value: log densities [n_t, n_y] at truth samples (from
    `truth_joint_logpdf` and `model_api.joint_log_density`) -> [n_t]."""
    return kl_mc(lp_truth, lp_model_joint)


def kl_truth_product_of_marginals(
    lp_truth: torch.Tensor,
    marginals: dict,
    rows: torch.Tensor,
) -> np.ndarray:
    """Monte Carlo KL(truth || product of the model's own marginals) over the
    non-intervened variables, per do-value -> [n_t].

    marginals: {variable j: (mu, sd, logw) [n_t, K]} for every
    non-intervened j (`model_api.joint_marginal_mixture`); rows: the truth
    samples [n_t, n_y, D] behind lp_truth [n_t, n_y].

    Subtracting `kl_truth_joint_model` gives what the joint head gains by
    modelling dependence: near zero on tasks with correlated outcomes means
    it learned none."""
    lp_product = sum(
        mix_logpdf(rows[..., j], mu, sd, logw)
        for j, (mu, sd, logw) in marginals.items()
    )
    return kl_mc(lp_truth, lp_product)


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Correlations [..., d, d] of covariances [..., d, d]."""
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return covariance / (sd[..., :, None] * sd[..., None, :])


def correlation_error(
    model_covariance: np.ndarray, truth_covariance: np.ndarray, T: int
) -> np.ndarray:
    """Mean absolute difference between the off-diagonal correlations of the
    model's mixture covariance [n_t, D, D] (`model_api.
    joint_mixture_mean_and_covariance`) and of the true covariance [D, D],
    over pairs of non-intervened variables, per do-value -> [n_t]. NaN when
    fewer than two variables are not intervened."""
    keep = _non_intervened(truth_covariance.shape[-1], T)
    if len(keep) < 2:
        return np.full(len(model_covariance), np.nan)
    upper = np.triu_indices(len(keep), k=1)
    model = correlation_matrix(model_covariance[:, keep][:, :, keep])
    truth = correlation_matrix(truth_covariance[np.ix_(keep, keep)])
    return np.abs(model[:, upper[0], upper[1]] - truth[upper]).mean(-1)


def mean_abs_correlation(covariance: np.ndarray, T: int) -> float:
    """Mean |off-diagonal correlation| of the non-intervened variables of a
    covariance [D, D] (how dependent the outcomes of do(X_T) are)."""
    keep = _non_intervened(covariance.shape[-1], T)
    if len(keep) < 2:
        return float("nan")
    upper = np.triu_indices(len(keep), k=1)
    corr = correlation_matrix(covariance[np.ix_(keep, keep)])
    return float(np.abs(corr[upper]).mean())


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
