"""Stub models and helpers for the causal_active_learning core tests (no CITNP).

`ConjugateStub` is a coherent Bayesian predictive with closed-form EIG, for
any D >= 3. Nodes: 0 treatment T, 1 node i, 2 outcome Y; others are noise.
Beliefs: one unknown coefficient beta ~ N(PRIOR_MEAN, PRIOR_VARIANCE) and
    Y | do(X_T = psi) ~ N(beta psi, NOISE).
A row informs beta through y = beta * z + e with z = strength[k] * v for a
row intervened on node k at value v (strength 0 for unlisted nodes), and
z = x_T for an observational row, x_T ~ N(0, OBSERVATIONAL_SCALE^2). Fantasies
are drawn row by row from the updated posterior (autoregressive), or all from
the current posterior (`independent_rows=True`, for comparison).
"""

import math

import numpy as np
import torch

from causal_active_learning.acquisitions import InterventionalQuery
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.value_search.domain import ValueDomain

DOUBLE = torch.float64
PSI = torch.linspace(-2, 2, 7, dtype=DOUBLE)
TREATMENT, INFORMATIVE, OUTCOME, UNINFORMATIVE = 0, 1, 2, 3


def query(D=4) -> InterventionalQuery:
    return InterventionalQuery(TREATMENT, OUTCOME, PSI)


class ConjugateStub:
    PRIOR_MEAN, PRIOR_VARIANCE, NOISE = 0.5, 1.0, 0.25

    def __init__(
        self,
        strength=None,
        observational=True,
        observational_scale=1.0,
        independent_rows=False,
    ):
        self.strength = {INFORMATIVE: 1.0} if strength is None else dict(strength)
        self.supports_observational_prediction = observational
        self.observational_scale = observational_scale
        self.independent_rows = independent_rows
        self.noise_draws = 0

    def _covariate(self, values, intervened):
        """z of every row [N] (0 for uninformative rows)."""
        z = torch.zeros(values.shape[0], dtype=DOUBLE)
        observational = ~intervened.any(-1)
        z[observational] = values[observational, TREATMENT]
        for node, c in self.strength.items():
            if node < values.shape[1]:
                rows = intervened[:, node]
                z[rows] = c * values[rows, node]
        return z

    def posterior(self, context: ExperimentalContext):
        z = self._covariate(context.values, context.intervened)
        y = context.values[:, OUTCOME]
        precision = 1 / self.PRIOR_VARIANCE + float((z**2).sum()) / self.NOISE
        mean = (
            self.PRIOR_MEAN / self.PRIOR_VARIANCE
            + float((z * y).sum()) / self.NOISE
        ) / precision
        return mean, 1 / precision

    def predict_query(self, contexts, treatment, outcome, treatment_values):
        psi = torch.as_tensor(treatment_values, dtype=DOUBLE)
        posts = [self.posterior(c) for c in contexts]
        m = torch.tensor([p[0] for p in posts], dtype=DOUBLE)
        s2 = torch.tensor([p[1] for p in posts], dtype=DOUBLE)
        return UnivariateGaussianMixture.gaussian(
            m[:, None] * psi, torch.sqrt(s2[:, None] * psi**2 + self.NOISE)
        )

    def draw_fantasy_noise(
        self, num_variables, num_fantasies, num_rows, generator
    ):
        self.noise_draws += 1
        return torch.randn(
            num_fantasies,
            num_rows,
            num_variables,
            generator=generator,
            dtype=DOUBLE,
        )

    def sample_fantasies(self, contexts, acquisitions, noise):
        if any(a.is_observational for a in acquisitions) and not (
            self.supports_observational_prediction
        ):
            raise NotImplementedError("no observational prediction")
        out = []
        for context, a, z in zip(contexts, acquisitions, noise):
            S, R = z.shape[0], a.num_rows
            m0, s20 = self.posterior(context)
            m = torch.full((S,), m0, dtype=DOUBLE)
            s2 = torch.full((S,), s20, dtype=DOUBLE)
            rows = z[:, :R].clone()
            for r in range(R):
                if a.is_observational:
                    covariate = self.observational_scale * z[:, r, TREATMENT]
                    rows[:, r, TREATMENT] = covariate
                else:
                    c = self.strength.get(a.target_node, 0.0)
                    covariate = torch.full(
                        (S,), c * a.intervention_value, dtype=DOUBLE
                    )
                y = (
                    m * covariate
                    + torch.sqrt(s2 * covariate**2 + self.NOISE)
                    * z[:, r, OUTCOME]
                )
                rows[:, r, OUTCOME] = y
                if a.target_node is not None:
                    rows[:, r, a.target_node] = a.intervention_value
                if not self.independent_rows:
                    precision = 1 / s2 + covariate**2 / self.NOISE
                    m = (m / s2 + covariate * y / self.NOISE) / precision
                    s2 = 1 / precision
            out.append(rows)
        return torch.stack(out)

    def exact_information_gain(self, node, v, psi=PSI, num_rows=1):
        """psi-averaged I(Y_psi; R outcomes of do(X_node = v)) under the prior:
        0.5 log(Var(Y_psi) / Var(Y_psi | data))."""
        c = self.strength.get(node, 0.0)
        return self._gain(num_rows * (c * v) ** 2, psi)

    def _gain(self, sum_z2, psi):
        s2, n = self.PRIOR_VARIANCE, self.NOISE
        s2_after = 1 / (1 / s2 + sum_z2 / n)
        psi = torch.as_tensor(psi, dtype=DOUBLE)
        return float(
            (0.5 * torch.log((s2 * psi**2 + n) / (s2_after * psi**2 + n))).mean()
        )

    def observational_information_gain(self, psi=PSI, num_rows=1, draws=400_000):
        """The same for R observational rows, by Monte Carlo over
        sum z^2 = scale^2 chi^2_R (exact up to MC error ~1e-4)."""
        rng = np.random.default_rng(0)
        sum_z2 = self.observational_scale**2 * rng.chisquare(num_rows, draws)
        s2, n = self.PRIOR_VARIANCE, self.NOISE
        psi = np.asarray(psi)
        s2_after = 1 / (1 / s2 + sum_z2[:, None] / n)
        gains = 0.5 * np.log((s2 * psi**2 + n) / (s2_after * psi**2 + n))
        return float(gains.mean())


class ContextBlindStub(ConjugateStub):
    """The query ignores the context: every EIG is zero."""

    def predict_query(self, contexts, treatment, outcome, treatment_values):
        psi = torch.as_tensor(treatment_values, dtype=DOUBLE)
        B = len(contexts)
        return UnivariateGaussianMixture.gaussian(
            torch.zeros(B, len(psi), dtype=DOUBLE),
            torch.ones(B, len(psi), dtype=DOUBLE),
        )


def empty_context(rows=20, D=4, seed=0):
    """An observational context of standard normal rows."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, D, generator=g, dtype=DOUBLE)
    return ExperimentalContext(values, torch.zeros(rows, D, dtype=torch.bool))


class StubEnvironment:
    """A linear Gaussian system on D >= 3 nodes (numpy): Y = X2 <- 0.8 X0 +
    0.8 X1; every other node independent noise."""

    def __init__(self, D=4):
        self.D = D
        B = np.zeros((D, D))
        B[0, 2], B[1, 2] = 0.8, 0.8
        self.B, self.noise = B, np.full(D, 0.3)
        self.order = [j for j in range(D) if j != 2] + [2]

    @property
    def num_variables(self):
        return self.D

    def _sample(self, n, node, value, rng):
        x = np.zeros((n, self.D))
        e = rng.standard_normal((n, self.D)) * np.sqrt(self.noise)
        for j in self.order:
            x[:, j] = value if j == node else x @ self.B[:, j] + e[:, j]
        return torch.as_tensor(x, dtype=DOUBLE)

    def sample_observational(self, num_rows, generator):
        return self._sample(num_rows, None, None, generator)

    def run(self, acquisition, generator):
        return self._sample(
            acquisition.num_rows,
            acquisition.target_node,
            acquisition.intervention_value,
            generator,
        )

    def true_query_distribution(self, treatment, outcome, values):
        psi = torch.as_tensor(values, dtype=DOUBLE)
        return UnivariateGaussianMixture.gaussian(
            0.8 * psi, torch.full_like(psi, math.sqrt(0.3 + 0.64 * 0.3))
        )


def value_key(value: float) -> int:
    return int(np.float64(value).view(np.int64)) & ((1 << 62) - 1)


def stub_evaluate(functions, standard_error, seed, calls=None):
    """evaluate(acquisitions) for EIG_node(u) = functions[node](u) + noise of
    the given standard error, deterministic per (seed, node, value)."""

    def evaluate(acquisitions):
        if calls is not None:
            calls.append([a.target_node for a in acquisitions])
        out = []
        for a in acquisitions:
            rng = np.random.default_rng(
                [seed, a.target_node, value_key(a.intervention_value)]
            )
            value = functions[a.target_node](a.intervention_value)
            value += standard_error * rng.standard_normal()
            out.append(
                InformationGainEstimate(
                    a, float(value), standard_error, torch.zeros(1)
                )
            )
        return out

    return evaluate


def domain(node, centre=0.0, half_width=1.0):
    return ValueDomain(node, centre, half_width)


def cheap_config(**kw):
    """A fast ActiveLearningConfig for tests."""
    from causal_active_learning.config import ActiveLearningConfig

    settings = dict(
        acquisition_budget=2,
        rows_per_acquisition=1,
        num_fantasies=8,
        final_num_fantasies=16,
        quadrature_points=256,
        bayesian_optimisation_iterations=2,
        gaussian_process_restarts=2,
        acquisition_grid_points=128,
    )
    settings.update(kw)
    return ActiveLearningConfig(**settings)


def prior_context(rows=20, D=4, seed=0):
    """A context that carries no information about beta for ConjugateStub
    (treatment column zero), so the posterior is the prior."""
    context = empty_context(rows, D, seed)
    values = context.values.clone()
    values[:, TREATMENT] = 0.0
    return ExperimentalContext(values, context.intervened)
