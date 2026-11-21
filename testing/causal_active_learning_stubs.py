"""Stub models and helpers for the causal_active_learning core tests (no CITNP).

`ConjugateStub` is a coherent Bayesian predictive with a closed-form EIG.
Nodes: 0 treatment T, 1 informative node i, 2 outcome Y, 3 uninformative node
j (D = 4; extra nodes, if D > 4, are pure noise). The outcome responds to the
treatment and to node i with the same unknown coefficient
beta ~ N(PRIOR_MEAN, PRIOR_VARIANCE), noise variance NOISE:
    Y | do(X_T = psi) ~ N(beta psi, NOISE).
Rows intervened on i observe y = beta v + e and update beta (conjugate);
nothing else does. Fantasies are drawn row by row from the updated posterior
(autoregressive), or all from the current posterior (`independent_rows`).
"""

import math

import numpy as np
import torch

from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.information_gain import InformationGainEstimate
from causal_active_learning.value_search.domain import ValueDomain

DOUBLE = torch.float64
PSI = torch.linspace(-2, 2, 7, dtype=DOUBLE)


class ConjugateStub:
    PRIOR_MEAN, PRIOR_VARIANCE, NOISE = 0.5, 1.0, 0.25
    INFORMATIVE, UNINFORMATIVE = 1, 3

    def __init__(self, independent_rows=False, observational=True):
        self.independent_rows = independent_rows
        self.supports_observational_prediction = observational
        self.query_calls: list[list[ExperimentalContext]] = []
        self.fantasy_calls: list[list[ExperimentalContext]] = []

    def posterior(self, context: ExperimentalContext):
        on_i = context.intervened[:, self.INFORMATIVE].to(DOUBLE)
        v, y = context.values[:, self.INFORMATIVE], context.values[:, 2]
        precision = (
            1 / self.PRIOR_VARIANCE + float((on_i * v**2).sum()) / self.NOISE
        )
        mean = (
            self.PRIOR_MEAN / self.PRIOR_VARIANCE
            + float((on_i * v * y).sum()) / self.NOISE
        ) / precision
        return mean, 1 / precision

    def predict_query(self, contexts, treatment, outcome, treatment_values):
        self.query_calls.append(list(contexts))
        psi = torch.as_tensor(treatment_values, dtype=DOUBLE)
        posts = [self.posterior(c) for c in contexts]
        m = torch.tensor([p[0] for p in posts], dtype=DOUBLE)
        s2 = torch.tensor([p[1] for p in posts], dtype=DOUBLE)
        mean = m[:, None] * psi
        sd = torch.sqrt(s2[:, None] * psi**2 + self.NOISE)
        return UnivariateGaussianMixture.gaussian(mean, sd)

    def draw_fantasy_noise(
        self, num_variables, num_fantasies, num_rows, generator
    ):
        return torch.randn(
            num_fantasies,
            num_rows,
            num_variables,
            generator=generator,
            dtype=DOUBLE,
        )

    def sample_fantasies(self, contexts, acquisitions, noise):
        self.fantasy_calls.append(list(contexts))
        out = []
        for context, a, z in zip(contexts, acquisitions, noise):
            S, R = z.shape[0], a.num_rows
            m0, s20 = self.posterior(context)
            m = torch.full((S,), m0, dtype=DOUBLE)
            s2 = torch.full((S,), s20, dtype=DOUBLE)
            rows = z[:, :R].clone()
            for r in range(R):
                if a.target_node == self.INFORMATIVE:
                    v = float(a.intervention_value)
                    y = m * v + torch.sqrt(s2 * v**2 + self.NOISE) * z[:, r, 2]
                    rows[:, r, 2] = y
                    if not self.independent_rows:
                        precision = 1 / s2 + v**2 / self.NOISE
                        m = (m / s2 + v * y / self.NOISE) / precision
                        s2 = 1 / precision
                if a.target_node is not None:
                    rows[:, r, a.target_node] = a.intervention_value
            out.append(rows)
        return torch.stack(out)

    def exact_information_gain(self, v, psi, num_rows=1):
        """psi-averaged I(Y_psi; R outcomes of do(X_i = v)) under the prior:
        0.5 log(Var(Y_psi) / Var(Y_psi | data))."""
        s2, n = self.PRIOR_VARIANCE, self.NOISE
        s2_after = 1 / (1 / s2 + num_rows * v**2 / n)
        psi = torch.as_tensor(psi, dtype=DOUBLE)
        return float(
            (0.5 * torch.log((s2 * psi**2 + n) / (s2_after * psi**2 + n))).mean()
        )


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
    """An observational context."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, D, generator=g, dtype=DOUBLE)
    return ExperimentalContext(values, torch.zeros(rows, D, dtype=torch.bool))


class StubEnvironment:
    """A linear Gaussian SCM on D = 4 nodes (numpy only), consistent with the
    stub's roles: X3 <- X4, Y = X2 <- 0.8 X0 + 0.8 X1."""

    def __init__(self):
        B = np.zeros((4, 4))
        B[0, 2], B[1, 2], B[3, 1] = 0.8, 0.8, 0.5
        self.B, self.noise = B, np.full(4, 0.3)

    @property
    def num_variables(self):
        return 4

    def _sample(self, n, node, value, rng):
        x = np.zeros((n, 4))
        e = rng.standard_normal((n, 4)) * np.sqrt(self.noise)
        for j in (0, 3, 1, 2):  # a topological order
            x[:, j] = value if j == node else x @ self.B[:, j] + e[:, j]
        return torch.as_tensor(x, dtype=DOUBLE)

    def sample_observational(self, num_rows, generator):
        return self._sample(num_rows, None, None, generator)

    def run(self, acquisition, generator):
        return self._sample(
            acquisition.num_rows, acquisition.target_node,
            acquisition.intervention_value, generator,
        )  # fmt: skip

    def true_query_distribution(self, treatment, outcome, values):
        psi = torch.as_tensor(values, dtype=DOUBLE)
        return UnivariateGaussianMixture.gaussian(
            0.8 * psi, torch.full_like(psi, math.sqrt(0.3))
        )


def value_key(value: float) -> int:
    return int(np.float64(value).view(np.int64)) & ((1 << 62) - 1)


def stub_evaluate(functions, standard_error, seed, calls=None):
    """evaluate(candidates) for EIG_node(u) = functions[node](u) + noise of
    the given standard error, deterministic per (seed, node, value)."""

    def evaluate(candidates):
        if calls is not None:
            calls.append([c.target_node for c in candidates])
        out = []
        for c in candidates:
            rng = np.random.default_rng(
                [seed, c.target_node, value_key(c.intervention_value)]
            )
            value = functions[c.target_node](c.intervention_value)
            value += standard_error * rng.standard_normal()
            out.append(
                InformationGainEstimate(
                    c, float(value), standard_error, torch.zeros(1)
                )
            )
        return out

    return evaluate


def domain(node, c=3.0):
    return ValueDomain(node, -c, c, centre=0.0, scale=1.0)
