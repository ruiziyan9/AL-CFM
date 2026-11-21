"""A small exact Gaussian process (GP) for the Bayesian optimisation surrogate
of EIG as a function of the intervention value (plan section 5.3).

- Targets are standardised by their mean and standard deviation before
  fitting and un-standardised for reporting.
- Constant mean; Matérn 5/2 kernel with one lengthscale and an output scale.
- Observation noise by policy (`fit_gaussian_process`):
  "standard_error": each point's variance is its squared standard error on
  the standardised target scale plus one fitted nugget in NUGGET_BOUNDS;
  "fitted": a fitted nugget only, bounded to [1e-6, mean squared standard
  error / variance of the estimates]; "fixed": 1e-6 only.
- Hyperparameters (lengthscale, output scale, constant mean, nugget) maximise
  the log marginal likelihood (L-BFGS in float64, log-parametrised, several
  random starts); lengthscale and nugget are bounded.
- `posterior` gives the latent function (without observation noise).

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
# Bounds of the nugget under the "standard_error" policy (standardised scale).
NUGGET_BOUNDS = (1e-6, 1e-2)
MINIMUM_NUGGET = 1e-6
NOISE_POLICIES = ("standard_error", "fitted", "fixed")
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

    nugget : float
        Homoscedastic noise variance added to every point (standardised
        scale).
    """

    lengthscale: float
    output_scale: float
    constant_mean: float
    nugget: float = 0.0


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

    matrix may carry leading batch axes ([..., n, n]); the jitter is shared.
    Returns (factor [..., n, n], the jitter that worked)."""
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype)
    current = float(jitter)
    for _ in range(retries + 1):
        factor, info = torch.linalg.cholesky_ex(matrix + current * identity)
        if not bool((info != 0).any()):
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
    - noise_variances: [n] float64, raw scale (the known part of the noise,
      squared standard errors; zero under the "fitted" and "fixed" policies).
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
    noise_policy: str = "given"
    nugget_bounds: tuple[float, float] | None = None

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
    factor, used = robust_cholesky(kernel + torch.diag(noise + h.nugget), jitter)
    residual = (y - h.constant_mean)[:, None]
    weights = torch.cholesky_solve(residual, factor)[:, 0]
    log_ml = (
        -0.5 * float(residual[:, 0] @ weights)
        - float(torch.log(torch.diagonal(factor)).sum())
        - 0.5 * len(y) * math.log(2 * math.pi)
    )
    return FittedGaussianProcess(
        inputs=inputs,
        targets=targets,
        noise_variances=noise_variances,
        target_mean=mean,
        target_scale=scale,
        hyperparameters=h,
        log_marginal_likelihood=log_ml,
        jitter=used,
        _cholesky=factor,
        _weights=weights,
    )


def _bounded(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    """exp(log lo + (log hi - log lo) sigmoid(raw)): log-scale, in bounds."""
    low, high = math.log(bounds[0]), math.log(bounds[1])
    return torch.exp(low + (high - low) * torch.sigmoid(raw))


def _distances(inputs: torch.Tensor) -> torch.Tensor:
    difference = inputs[:, None, :] - inputs[None, :, :]
    return torch.sqrt((difference**2).sum(-1) + 1e-30)


def _negative_log_marginal_likelihoods(
    raw, distances, y, noise, bounds, nugget_bounds, jitter
):
    """Negative log marginal likelihood [B] of B hyperparameter vectors
    raw [B, 4] (lengthscale, output scale, mean, nugget; bounded transforms)."""
    lengthscale = _bounded(raw[:, 0], bounds)[:, None, None]
    output_scale = _bounded(raw[:, 1], OUTPUT_SCALE_BOUNDS)[:, None, None]
    root5_r = math.sqrt(5.0) * distances / lengthscale  # [B, n, n]
    kernel = (
        output_scale * (1.0 + root5_r + root5_r**2 / 3.0) * torch.exp(-root5_r)
    )
    diagonal = noise.expand(raw.shape[0], -1)
    if nugget_bounds is not None:
        diagonal = diagonal + _bounded(raw[:, 3], nugget_bounds)[:, None]
    factor, _ = robust_cholesky(kernel + torch.diag_embed(diagonal), jitter)
    residual = (y - raw[:, 2:3])[..., None]  # [B, n, 1]
    weights = torch.cholesky_solve(residual, factor)
    return (
        0.5 * (residual * weights).sum((-2, -1))
        + torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
        + 0.5 * y.shape[-1] * math.log(2 * math.pi)
    )


class _SingleThread:
    """Run small linear algebra on one thread (many threads make the tiny GP
    matrices an order of magnitude slower); restores the setting."""

    def __enter__(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.threads)


def fit_gaussian_process(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_variances: torch.Tensor,
    lengthscale_bounds: tuple[float, float],
    generator: torch.Generator,
    restarts: int = 5,
    jitter: float = DEFAULT_JITTER,
    noise_policy: str = "standard_error",
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
        [n] raw squared standard errors; standardised with the targets.

    lengthscale_bounds : tuple[float, float]
        (lower, upper), in input units.

    generator : torch.Generator
        Source of the random starts.

    restarts : int
        Number of random starts (>= 1).

    jitter : float
        Diagonal jitter (doubled on Cholesky failure, `robust_cholesky`): a
        numerical safeguard, not a noise model.

    noise_policy : str
        "standard_error", "fitted" or "fixed" (see the module docstring).
    """
    inputs = torch.as_tensor(inputs, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    noise_variances = torch.as_tensor(noise_variances, dtype=torch.float64)
    if not 0 < lengthscale_bounds[0] < lengthscale_bounds[1]:
        raise ValueError("lengthscale_bounds must satisfy 0 < lower < upper")
    if noise_policy not in NOISE_POLICIES:
        raise ValueError(f"noise_policy must be one of {NOISE_POLICIES}")
    mean, scale = _standardise(targets)
    y = (targets - mean) / scale
    standardised_variances = noise_variances / scale**2
    nugget_bounds: tuple[float, float] | None
    if noise_policy == "standard_error":
        noise, nugget_bounds = standardised_variances, NUGGET_BOUNDS
    elif noise_policy == "fitted":
        upper = float(standardised_variances.mean()) if len(y) else 0.0
        noise = torch.zeros_like(y)
        nugget_bounds = (MINIMUM_NUGGET, max(upper, 2 * MINIMUM_NUGGET))
    else:  # fixed
        noise, nugget_bounds = torch.full_like(y, MINIMUM_NUGGET), None
    known = noise * scale**2  # the known noise on the raw scale
    starts = torch.rand(restarts, 3, generator=generator, dtype=torch.float64)
    starts = starts.clamp(1e-3, 1 - 1e-3)
    logits = torch.log(starts / (1 - starts))
    raw = torch.stack(
        [
            logits[:, 0],
            logits[:, 1],
            torch.zeros(restarts, dtype=torch.float64),
            logits[:, 2],
        ],
        dim=-1,
    ).requires_grad_(True)
    distances = _distances(inputs)

    def losses(parameters):
        return _negative_log_marginal_likelihoods(
            parameters,
            distances,
            y,
            noise,
            lengthscale_bounds,
            nugget_bounds,
            jitter,
        )

    # The restarts are independent, so their sum is minimised jointly (one
    # batched L-BFGS run); the best restart is kept afterwards.
    optimiser = torch.optim.LBFGS(
        [raw], max_iter=_LBFGS_ITERATIONS, line_search_fn="strong_wolfe"
    )

    def closure():
        optimiser.zero_grad()
        loss = losses(raw).sum()
        loss.backward()
        return loss

    with _SingleThread():
        try:
            optimiser.step(closure)
        except torch.linalg.LinAlgError:
            pass
        with torch.no_grad():
            try:
                final = losses(raw.detach())
            except torch.linalg.LinAlgError:
                final = torch.full((restarts,), math.inf, dtype=torch.float64)
    final = torch.where(torch.isfinite(final), final, torch.inf)
    if not bool(torch.isfinite(final).any()):
        raise torch.linalg.LinAlgError("every GP hyperparameter fit failed")
    best = (float(final.min()), raw.detach()[int(torch.argmin(final))])
    raw = best[1]
    hyperparameters = GaussianProcessHyperparameters(
        lengthscale=float(_bounded(raw[0], lengthscale_bounds)),
        output_scale=float(_bounded(raw[1], OUTPUT_SCALE_BOUNDS)),
        constant_mean=float(raw[2]),
        nugget=0.0
        if nugget_bounds is None
        else float(_bounded(raw[3], nugget_bounds)),
    )
    fitted = condition_gaussian_process(
        inputs, targets, known, hyperparameters, jitter
    )
    fitted.noise_policy = noise_policy
    fitted.nugget_bounds = nugget_bounds
    return fitted
