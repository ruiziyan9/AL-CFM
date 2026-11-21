"""A small exact Gaussian process (GP) for the Bayesian optimisation surrogate
of EIG as a function of the intervention value (plan section 5.3).

- Targets are standardised by their mean and standard deviation before
  fitting and un-standardised for reporting.
- Constant mean; Matérn 5/2 kernel with one lengthscale and an output scale.
- Known heteroscedastic observation noise: each point's noise variance is
  given (the squared standard error of its EIG estimate, on the standardised
  target scale) plus a jitter; it is not fitted.
- Hyperparameters (lengthscale, output scale, constant mean) maximise the log
  marginal likelihood (L-BFGS in float64, log-parametrised, several random
  starts); the lengthscale is bounded.

Inputs have shape [num_points, value_dimension] so that multi-node
interventions can reuse the code later; the kernel is isotropic.
"""

import math
from dataclasses import dataclass

import torch

DEFAULT_JITTER = 1e-6
CHOLESKY_RETRIES = 5
# Bounds of the output scale (a variance) on the standardised target scale.
OUTPUT_SCALE_BOUNDS = (1e-3, 1e2)
_LBFGS_ITERATIONS = 100


@dataclass(frozen=True)
class GaussianProcessHyperparameters:
    """Kernel and mean hyperparameters on the standardised target scale.

    Attributes:
    ----------
    lengthscale : float
        Matérn 5/2 lengthscale, in input units.

    output_scale : float
        Kernel variance.

    constant_mean : float
        Prior mean.
    """

    lengthscale: float
    output_scale: float
    constant_mean: float


def matern52_kernel(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor | float,
    output_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Matérn 5/2 kernel matrix [n, m] of inputs [n, d] and [m, d]:
    s (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r), r = |a - b| / lengthscale."""
    difference = first[:, None, :] - second[None, :, :]
    r = torch.sqrt((difference**2).sum(-1) + 1e-30) / lengthscale
    root5_r = math.sqrt(5.0) * r
    return output_scale * (1.0 + root5_r + root5_r**2 / 3.0) * torch.exp(-root5_r)


def robust_cholesky(
    matrix: torch.Tensor,
    jitter: float = DEFAULT_JITTER,
    retries: int = CHOLESKY_RETRIES,
) -> tuple[torch.Tensor, float]:
    """Lower Cholesky factor of matrix + jitter I with
    `torch.linalg.cholesky_ex`; on failure the jitter is doubled and the
    factorisation retried up to `retries` times, then a
    torch.linalg.LinAlgError is raised.

    Returns (factor [n, n], the jitter that worked)."""
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype)
    current = float(jitter)
    for _ in range(retries + 1):
        factor, info = torch.linalg.cholesky_ex(matrix + current * identity)
        if int(info) == 0:
            return factor, current
        current *= 2.0
    raise torch.linalg.LinAlgError(
        f"Cholesky failed with jitter up to {current / 2.0:.3g} "
        f"({retries} retries)"
    )


@dataclass
class FittedGaussianProcess:
    """A GP conditioned on its training data, ready for posterior queries.

    Attributes:
    -----------
    - inputs: [n, d] float64.
    - targets: [n] float64, raw scale.
    - noise_variances: [n] float64, raw scale (squared standard errors).
    - target_mean, target_scale: the standardisation of the targets.
    - hyperparameters: on the standardised scale.
    - log_marginal_likelihood: of the standardised targets.
    - jitter: the diagonal jitter that was used.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    noise_variances: torch.Tensor
    target_mean: float
    target_scale: float
    hyperparameters: GaussianProcessHyperparameters
    log_marginal_likelihood: float
    jitter: float
    _cholesky: torch.Tensor
    _weights: torch.Tensor  # (K + N)^-1 (y - m), standardised

    def posterior(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and standard deviation [m] of the latent function
        at points [m, d], on the raw target scale (no observation noise)."""
        points = torch.as_tensor(points, dtype=torch.float64)
        h = self.hyperparameters
        cross = matern52_kernel(
            self.inputs, points, h.lengthscale, h.output_scale
        )
        mean = h.constant_mean + cross.T @ self._weights
        solved = torch.linalg.solve_triangular(self._cholesky, cross, upper=False)
        variance = (h.output_scale - (solved**2).sum(0)).clamp(min=0.0)
        return (
            self.target_mean + self.target_scale * mean,
            self.target_scale * variance.sqrt(),
        )


def _standardise(targets: torch.Tensor) -> tuple[float, float]:
    mean = float(targets.mean())
    scale = float(targets.std(unbiased=False)) if len(targets) > 1 else 0.0
    return mean, (scale if scale > 1e-12 else 1.0)


def condition_gaussian_process(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_variances: torch.Tensor,
    hyperparameters: GaussianProcessHyperparameters,
    jitter: float = DEFAULT_JITTER,
) -> FittedGaussianProcess:
    """The GP with the given hyperparameters (standardised scale) conditioned
    on inputs [n, d], raw targets [n] and raw noise variances [n]."""
    inputs = torch.as_tensor(inputs, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    noise_variances = torch.as_tensor(noise_variances, dtype=torch.float64)
    mean, scale = _standardise(targets)
    y = (targets - mean) / scale
    noise = noise_variances / scale**2
    h = hyperparameters
    kernel = matern52_kernel(inputs, inputs, h.lengthscale, h.output_scale)
    factor, used = robust_cholesky(kernel + torch.diag(noise), jitter)
    residual = (y - h.constant_mean)[:, None]
    weights = torch.cholesky_solve(residual, factor)[:, 0]
    log_ml = (
        -0.5 * float(residual[:, 0] @ weights)
        - float(torch.log(torch.diagonal(factor)).sum())
        - 0.5 * len(y) * math.log(2 * math.pi)
    )
    return FittedGaussianProcess(
        inputs=inputs, targets=targets, noise_variances=noise_variances,
        target_mean=mean, target_scale=scale, hyperparameters=h,
        log_marginal_likelihood=log_ml, jitter=used, _cholesky=factor,
        _weights=weights,
    )  # fmt: skip


def _bounded(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    """exp(log lo + (log hi - log lo) sigmoid(raw)): log-scale, in bounds."""
    low, high = math.log(bounds[0]), math.log(bounds[1])
    return torch.exp(low + (high - low) * torch.sigmoid(raw))


def _negative_log_marginal_likelihood(raw, inputs, y, noise, bounds, jitter):
    lengthscale = _bounded(raw[0], bounds)
    output_scale = _bounded(raw[1], OUTPUT_SCALE_BOUNDS)
    kernel = matern52_kernel(inputs, inputs, lengthscale, output_scale)
    factor, _ = robust_cholesky(kernel + torch.diag(noise), jitter)
    residual = (y - raw[2])[:, None]
    weights = torch.cholesky_solve(residual, factor)
    return (
        0.5 * (residual * weights).sum()
        + torch.log(torch.diagonal(factor)).sum()
        + 0.5 * len(y) * math.log(2 * math.pi)
    )


def fit_gaussian_process(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_variances: torch.Tensor,
    lengthscale_bounds: tuple[float, float],
    generator: torch.Generator,
    restarts: int = 5,
    jitter: float = DEFAULT_JITTER,
) -> FittedGaussianProcess:
    """Fit the hyperparameters by maximising the log marginal likelihood and
    condition the GP on the data.

    Each of `restarts` L-BFGS runs (float64, strong-Wolfe line search)
    starts from a random point drawn from `generator`: lengthscale and
    output scale log-uniform within their bounds, constant mean 0. The
    lengthscale stays within `lengthscale_bounds` and the output scale within
    OUTPUT_SCALE_BOUNDS through a sigmoid in log space. The run with the
    largest log marginal likelihood is kept.

    Args:
    ----------
    inputs : torch.Tensor
        [n, d].

    targets : torch.Tensor
        [n] raw (e.g. EIG estimates).

    noise_variances : torch.Tensor
        [n] raw (squared standard errors); standardised with the targets.

    lengthscale_bounds : tuple[float, float]
        (lower, upper), in input units.

    generator : torch.Generator
        Source of the random starts.

    restarts : int
        Number of random starts (>= 1).

    jitter : float
        Diagonal jitter (doubled on Cholesky failure, `robust_cholesky`).
    """
    inputs = torch.as_tensor(inputs, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    noise_variances = torch.as_tensor(noise_variances, dtype=torch.float64)
    if not 0 < lengthscale_bounds[0] < lengthscale_bounds[1]:
        raise ValueError("lengthscale_bounds must satisfy 0 < lower < upper")
    mean, scale = _standardise(targets)
    y = (targets - mean) / scale
    noise = noise_variances / scale**2
    starts = torch.rand(restarts, 2, generator=generator, dtype=torch.float64)
    starts = starts.clamp(1e-3, 1 - 1e-3)
    best: tuple[float, torch.Tensor] | None = None
    for start in starts:
        logit = torch.log(start / (1 - start))
        raw = torch.cat([logit, torch.zeros(1, dtype=torch.float64)])
        raw.requires_grad_(True)
        optimiser = torch.optim.LBFGS(
            [raw], max_iter=_LBFGS_ITERATIONS, line_search_fn="strong_wolfe"
        )

        def closure():
            optimiser.zero_grad()
            loss = _negative_log_marginal_likelihood(
                raw, inputs, y, noise, lengthscale_bounds, jitter
            )
            loss.backward()
            return loss

        try:
            optimiser.step(closure)
            with torch.no_grad():
                loss = float(
                    _negative_log_marginal_likelihood(
                        raw, inputs, y, noise, lengthscale_bounds, jitter
                    )
                )
        except torch.linalg.LinAlgError:
            continue
        if math.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, raw.detach().clone())
    if best is None:
        raise torch.linalg.LinAlgError("every GP hyperparameter fit failed")
    raw = best[1]
    hyperparameters = GaussianProcessHyperparameters(
        lengthscale=float(_bounded(raw[0], lengthscale_bounds)),
        output_scale=float(_bounded(raw[1], OUTPUT_SCALE_BOUNDS)),
        constant_mean=float(raw[2]),
    )
    return condition_gaussian_process(
        inputs, targets, noise_variances, hyperparameters, jitter
    )
