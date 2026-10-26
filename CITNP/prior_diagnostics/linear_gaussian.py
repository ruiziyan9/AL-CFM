"""
Exact distributions of a generated linear-Gaussian task.

The `linearnonidentifiable` family is a linear-Gaussian SCM (structural causal
model). After the reference standardisation each node is

    X_j = (w_j . X_pa(j) + eps_j - loc_j) / scale_j,   eps_j ~ N(0, s_j^2)

with s_j^2 = var1 (nodes with parents) or var2 (roots). So in the task's units
X = c + B^T X + e with B[i, j] = w_ij / scale_j, c_j = -loc_j / scale_j and
Var(e_j) = s_j^2 / scale_j^2. Both the observational and the interventional
distributions are then Gaussian with closed-form moments.

Needs generate_task(..., return_debug=True) with the mechanism record (see
generator_patch_3.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CITNP.datasets.functions_generator import LINEAR_NIG_MECHANISMS

__all__ = [
    "LinearGaussianSCM",
    "gauss_kl",
    "population_covariance",
    "scm_from_debug",
]


@dataclass
class LinearGaussianSCM:
    """X = c + B^T X + e, e ~ N(0, diag(omega)); B[i, j] = coef of X_i in X_j."""

    B: np.ndarray
    c: np.ndarray
    omega: np.ndarray

    def moments(self, do: dict[int, float] | None = None):
        """Mean and covariance of X, optionally under do(X_t = x) for each
        (t, x) in `do` (incoming edges cut, constant value, no noise)."""
        B, c, w = self.B.copy(), self.c.copy(), self.omega.copy()
        for t, x in (do or {}).items():
            B[:, t] = 0.0
            c[t] = x
            w[t] = 0.0
        A = np.linalg.inv(np.eye(len(c)) - B.T)
        return A @ c, A @ np.diag(w) @ A.T

    def conditional_line(self, t: int, y: int):
        """Y | X_t = x ~ N(a + b x, v): returns (a, b, v)."""
        mu, S = self.moments()
        b = S[y, t] / S[t, t]
        return mu[y] - b * mu[t], b, S[y, y] - b * S[y, t]

    def interventional_line(self, t: int, y: int):
        """Y | do(X_t = x) ~ N(a + b x, v): returns (a, b, v)."""
        m0, V0 = self.moments({t: 0.0})
        m1, _ = self.moments({t: 1.0})
        return m0[y], m1[y] - m0[y], V0[y, y]


def gauss_kl(m1, v1, m0, v0):
    """KL(N(m1, v1) || N(m0, v0)), elementwise."""
    return 0.5 * (np.log(v0 / v1) + (v1 + (m1 - m0) ** 2) / v0 - 1.0)


def scm_from_debug(debug: dict) -> LinearGaussianSCM:
    """Linear-Gaussian SCM of a task from generate_task(..., return_debug=True),
    in the task's (permuted) variable indices."""
    mech = debug["mechanisms"]
    D = len(mech)
    B, c, w = np.zeros((D, D)), np.zeros(D), np.zeros(D)
    for j, rec in mech.items():
        f = rec["mechanism"]
        if not isinstance(f, LINEAR_NIG_MECHANISMS):
            raise NotImplementedError(
                f"exact moments need a linear-Gaussian NIG family, got {type(f).__name__}"
            )
        s, pa = rec["scale"], rec["parents"]
        if pa:
            B[pa, j] = np.asarray(f.weight).ravel() / s
            w[j] = float(np.ravel(f.var1)[0]) / s**2
        else:
            w[j] = float(np.ravel(f.var2)[0]) / s**2
        c[j] = -rec["loc"] / s
    p = np.asarray(debug["perm"])  # new index k holds old variable p[k]
    return LinearGaussianSCM(B[p][:, p], c[p], w[p])


def population_covariance(graph: np.ndarray, mechanisms: dict) -> np.ndarray:
    """Population covariance (I - B^T)^-1 diag(s2) (I - B^T)^-T of the SCM
    defined by the sampled weights and noise variances, in raw units (no
    reference standardisation, no intervention).

    Args:
    ----------
    graph : np.ndarray
        [D, D] DAG with graph[i, j] = 1 iff i -> j.

    mechanisms : dict
        node -> {"parents", "mechanism"} as in generate_task(...,
        return_debug=True)["mechanisms"], or node -> mechanism object. Each
        mechanism must have been called once (so its weights are drawn).
    """
    D = graph.shape[0]
    B, s2 = np.zeros((D, D)), np.zeros(D)
    for j in range(D):
        rec = mechanisms[j]
        f = rec["mechanism"] if isinstance(rec, dict) else rec
        if not isinstance(f, LINEAR_NIG_MECHANISMS):
            raise NotImplementedError(
                f"needs a linear-Gaussian NIG family, got {type(f).__name__}"
            )
        pa = np.where(graph[:, j])[0]
        if len(pa):
            B[pa, j] = np.asarray(f.weight).ravel()
        s2[j] = float(np.ravel(f.var1 if len(pa) else f.var2)[0])
    A = np.linalg.inv(np.eye(D) - B.T)
    return A @ np.diag(s2) @ A.T
