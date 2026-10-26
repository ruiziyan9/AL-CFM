"""
Reference answers: the exact Bayesian oracle and known-truth linear SCMs.

Exact Bayesian oracle for small linear-Gaussian SCMs under the
linear-Gaussian NIG priors (`linearnonidentifiable`, `linear_nig_legacy`,
`linear_nig`, `linear_iw`) with normalise=False.

Prior (a `LinearGaussianPrior`; matches functions_generator.LinearNIG and
LinearNIGLegacy), per node j with p parents:
    x_j = w_j^T x_pa + e_j, e_j ~ N(0, s2),  s2 ~ IG(prior.node_shape(p), ig_scale),
               w_j | s2 ~ N(0, eta * s2 * I), eta = prior.eta
Legacy prior: shape 2.5 for roots and 3.0 for every non-root, ig_scale 0.5;
IW prior: shape alpha0 + p / 2.
This is Normal-Inverse-Gamma conjugate, so every DAG's marginal likelihood is
closed form. With weight_mean = m != 0 (signed-mean prior) the weight prior is
    w_j | s2 ~ sum_s 2^-p N(m * s, eta * s2 * I),  s in {+1, -1}^p,
a mixture of 2^p NIG components, which stays exact in closed form.
Hard interventions are handled by dropping node j's likelihood
term on rows where j was intervened (its value is set, not generated); the
intervened values still act as parents of j's children.

The posterior predictive for p(X_Y | do(X_T = t), data) is
    sum_G P(G | data) * E_{theta ~ p(theta | G, data)} N(y; mu_theta(t), v_theta)
with the inner expectation by Monte Carlo over exact conjugate posterior draws
(each draw gives a Gaussian interventional distribution in closed form).

numpy / scipy only; no torch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from CITNP.datasets.linear_gaussian_prior import LinearGaussianPrior

# Deprecated: the legacy prior constants. The hyperparameters now live in
# `LinearGaussianPrior`; these are kept (and re-exported by oracle_lingauss.py)
# for backward compatibility only.
ROOT_A, ROOT_B = 2.5, 0.5
NODE_A, NODE_B = 3.0, 0.5
ETA = 1.0


def _resolve_prior(
    prior: LinearGaussianPrior | None, eta: float, weight_mean: float
) -> LinearGaussianPrior:
    """`prior` if given, else the legacy prior with the given eta / weight_mean."""
    if prior is not None:
        return prior
    return LinearGaussianPrior.legacy(eta=eta, weight_mean=weight_mean)


# --------------------------------------------------------------------------- #
# DAG enumeration
# --------------------------------------------------------------------------- #
def is_acyclic(adj: np.ndarray) -> bool:
    d = adj.shape[0]
    m = adj.astype(int).copy()
    reach = m.copy()
    for _ in range(d):
        reach = ((reach + reach @ m) > 0).astype(int)
    return not np.any(np.diag(reach))


def all_dags(d: int) -> list[np.ndarray]:
    """All labelled DAGs on d nodes (25 for d = 3). adj[i, j] = 1 means i -> j."""
    pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    out = []
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        adj = np.zeros((d, d), dtype=int)
        for (i, j), b in zip(pairs, bits):
            adj[i, j] = b
        if is_acyclic(adj):
            out.append(adj)
    return out


def has_path(adj: np.ndarray, i: int, j: int) -> bool:
    d = adj.shape[0]
    reach = np.linalg.matrix_power(np.eye(d, dtype=int) + adj, d)
    return bool(reach[i, j] > 0) and i != j


def dag_key(adj: np.ndarray) -> tuple:
    return tuple(adj.flatten().tolist())


def skeleton(adj: np.ndarray) -> np.ndarray:
    """Undirected skeleton of a DAG: symmetric 0/1 matrix, 1 iff i and j are adjacent."""
    a = np.asarray(adj) != 0
    return (a | a.T).astype(int)


def v_structures(adj: np.ndarray) -> set:
    """Set of (a, c, b), a < b, with a -> c <- b and a, b not adjacent."""
    a = np.asarray(adj) != 0
    skel = skeleton(adj)
    out = set()
    for c in range(a.shape[0]):
        pa = np.where(a[:, c])[0]
        for i, p in enumerate(pa):
            for q in pa[i + 1 :]:
                if not skel[p, q]:
                    out.add((int(p), int(c), int(q)))
    return out


def mec_key(adj: np.ndarray) -> tuple:
    """Hashable identifier of the Markov equivalence class (MEC) of a DAG:
    skeleton plus v-structures."""
    return (dag_key(skeleton(adj)), tuple(sorted(v_structures(adj))))


# --------------------------------------------------------------------------- #
# Conjugate node model
# --------------------------------------------------------------------------- #
@dataclass
class NodePosterior:
    root: bool
    a_n: float
    b_n: float
    m_n: np.ndarray | None  # posterior mean of weights
    V_n: np.ndarray | None  # posterior cov / s2 of weights
    log_ml: float
    # Signed-mean mixture prior only (weight_mean != 0), one entry per sign
    # pattern s; None otherwise. m_n / b_n above then hold component 0.
    comp_m: np.ndarray | None = None  # [S, p] posterior means
    comp_b: np.ndarray | None = None  # [S] posterior IG scales
    comp_logw: np.ndarray | None = None  # [S] normalised log posterior weights


def node_posterior(
    x: np.ndarray,
    int_mask: np.ndarray,
    j: int,
    parents: np.ndarray,
    eta: float = ETA,
    weight_mean: float = 0.0,
    prior: LinearGaussianPrior | None = None,
) -> NodePosterior:
    """NIG posterior and log marginal likelihood for node j given its parents,
    using only rows where j was NOT intervened.

    With weight_mean = m != 0 the weight prior is the signed-mean mixture
    sum_s 2^-p N(m * s, eta * s2 * I) over sign patterns s in {+1, -1}^p: one
    NIG component per pattern, so the posterior is a mixture of NIGs.

    `prior` gives the IG shape (`prior.node_shape(len(parents))`), IG scale, eta
    and weight_mean; without it the legacy prior with `eta` / `weight_mean` is
    used."""
    prior = _resolve_prior(prior, eta, weight_mean)
    eta, weight_mean = prior.eta, prior.weight_mean
    keep = ~int_mask[:, j]
    y = x[keep, j]
    n = y.shape[0]
    if len(parents) == 0:
        a, b = prior.node_shape(0), prior.ig_scale
        a_n = a + n / 2
        b_n = b + 0.5 * float(y @ y)
        log_ml = (
            gammaln(a_n)
            - gammaln(a)
            + a * np.log(b)
            - a_n * np.log(b_n)
            - 0.5 * n * np.log(2 * np.pi)
        )
        return NodePosterior(True, a_n, b_n, None, None, float(log_ml))

    a, b = prior.node_shape(len(parents)), prior.ig_scale
    X = x[keep][:, parents]
    p = X.shape[1]
    prec = X.T @ X + np.eye(p) / eta
    V_n = np.linalg.inv(prec)
    m_n = V_n @ X.T @ y
    a_n = a + n / 2
    b_n = b + 0.5 * float(y @ y - m_n @ prec @ m_n)
    _, logdet = np.linalg.slogdet(np.eye(p) + eta * X.T @ X)
    if weight_mean != 0.0:
        # One NIG component per sign pattern s, prior mean mu_s = m * s. All
        # components share prec, V_n, a_n and logdet; mean and b differ.
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=p)))  # [S, p]
        mu = weight_mean * signs
        comp_m = (X.T @ y + mu / eta) @ V_n.T  # rows: V_n (X^T y + mu_s / eta)
        comp_b = b + 0.5 * (
            float(y @ y)
            + (mu * mu).sum(1) / eta
            - np.einsum("sp,pq,sq->s", comp_m, prec, comp_m)
        )
        log_ml_s = (
            gammaln(a_n)
            - gammaln(a)
            + a * np.log(b)
            - a_n * np.log(comp_b)
            - 0.5 * n * np.log(2 * np.pi)
            - 0.5 * logdet
        )
        lse = logsumexp(log_ml_s)
        return NodePosterior(
            False,
            a_n,
            float(comp_b[0]),
            comp_m[0],
            V_n,
            float(lse - p * np.log(2.0)),
            comp_m=comp_m,
            comp_b=comp_b,
            comp_logw=log_ml_s - lse,
        )
    log_ml = (
        gammaln(a_n)
        - gammaln(a)
        + a * np.log(b)
        - a_n * np.log(b_n)
        - 0.5 * n * np.log(2 * np.pi)
        - 0.5 * logdet
    )
    return NodePosterior(False, a_n, b_n, m_n, V_n, float(log_ml))


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
class LinGaussOracle:
    """Exact posterior over DAGs + MC posterior predictive for do-queries.

    Args:
        log_graph_prior: dict dag_key -> log prior. Use `empirical_graph_prior`
            (matches the training generator) or None for uniform over DAGs.
        eta: weight-prior scale; must equal the data config's weight_scale.
        weight_mean: signed weight-prior mean; must equal the data config's
            weight_mean (0 = the original zero-mean prior).
        prior: `LinearGaussianPrior` of the data config (`LinearGaussianPrior.from_config`);
            overrides eta and weight_mean when given.
    """

    def __init__(
        self,
        d: int,
        log_graph_prior: dict | None = None,
        eta: float = ETA,
        weight_mean: float = 0.0,
        prior: LinearGaussianPrior | None = None,
    ):
        self.d = d
        self.prior = _resolve_prior(prior, eta, weight_mean)
        self.eta = self.prior.eta
        self.weight_mean = self.prior.weight_mean
        self.dags = all_dags(d)
        if log_graph_prior is None:
            lp = np.full(len(self.dags), -np.log(len(self.dags)))
        else:
            lp = np.array(
                [log_graph_prior.get(dag_key(g), -np.inf) for g in self.dags]
            )
        self.log_prior = lp

    def fit(self, x: np.ndarray, int_mask: np.ndarray):
        """x [n, d] raw (unnormalised) context, int_mask [n, d] bool."""
        self.node_post = []
        log_ml = np.zeros(len(self.dags))
        cache: dict = {}
        for g_i, g in enumerate(self.dags):
            posts = []
            for j in range(self.d):
                pa = tuple(np.where(g[:, j])[0].tolist())
                key = (j, pa)
                if key not in cache:
                    cache[key] = node_posterior(
                        x,
                        int_mask,
                        j,
                        np.array(pa, dtype=int),
                        prior=self.prior,
                    )
                posts.append(cache[key])
                log_ml[g_i] += cache[key].log_ml
            self.node_post.append(posts)
        lj = self.log_prior + log_ml
        self.log_post = lj - logsumexp(lj)
        return self

    @property
    def graph_post(self) -> np.ndarray:
        return np.exp(self.log_post)

    def prob_ancestor(self, i: int, j: int) -> float:
        """P(i is a causal ancestor of j | data)."""
        return float(
            sum(
                p for p, g in zip(self.graph_post, self.dags) if has_path(g, i, j)
            )
        )

    def graph_entropy(self) -> float:
        p = self.graph_post
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _sample_params(self, g_i: int, n_mc: int, rng):
        """Weight matrix B [n_mc, d, d] (B[:, i, j] weight i->j) and noise s2 [n_mc, d]."""
        g = self.dags[g_i]
        B = np.zeros((n_mc, self.d, self.d))
        s2 = np.zeros((n_mc, self.d))
        for j, post in enumerate(self.node_post[g_i]):
            if post.comp_m is None:  # single NIG (roots: m_n is None)
                b_n = post.b_n
                m_n = None if post.m_n is None else post.m_n[None]
            else:
                # Signed-mean mixture: one component (sign pattern) per draw.
                pk = np.exp(post.comp_logw)
                k = rng.choice(len(pk), size=n_mc, p=pk / pk.sum())
                b_n, m_n = post.comp_b[k], post.comp_m[k]  # [n_mc], [n_mc, p]
            s2_j = 1.0 / rng.gamma(post.a_n, 1.0 / b_n, size=n_mc)  # IG(a_n, b_n)
            s2[:, j] = s2_j
            if not post.root:
                pa = np.where(g[:, j])[0]
                L = np.linalg.cholesky(post.V_n)
                z = rng.standard_normal((n_mc, len(pa)))
                w = m_n + np.sqrt(s2_j)[:, None] * (z @ L.T)
                B[:, pa, j] = w
        return B, s2

    def predictive_components(
        self,
        T: int,
        Y: int,
        n_mc: int = 2000,
        min_graph_prob: float = 1e-6,
        seed: int = 0,
    ):
        """Mixture representation of p(X_Y | do(X_T = t), data):
        returns (log_w [M], slope [M], var [M]) with component mean = slope * t.
        (Zero-mean priors and linear SEM => interventional mean is linear in t.)
        """
        rng = np.random.default_rng(seed)
        log_ws: list[np.ndarray] = []
        slopes: list[np.ndarray] = []
        vars_: list[np.ndarray] = []
        for g_i, lp in enumerate(self.log_post):
            if np.exp(lp) < min_graph_prob:
                continue
            B, s2 = self._sample_params(g_i, n_mc, rng)
            B[:, :, T] = 0.0  # cut edges into T
            s2 = s2.copy()
            s2[:, T] = 0.0  # T is fixed
            # x = (I - B^T)^{-1} (e + t u_T)
            A = np.linalg.inv(np.eye(self.d)[None] - np.transpose(B, (0, 2, 1)))
            slopes.append(A[:, Y, T])
            vars_.append(np.einsum("mk,mk->m", A[:, Y, :] ** 2, s2))
            log_ws.append(np.full(n_mc, lp - np.log(n_mc)))
        return (
            np.concatenate(log_ws),
            np.concatenate(slopes),
            np.concatenate(vars_),
        )

    def logpdf(
        self, T: int, Y: int, t: np.ndarray, y: np.ndarray, **kw
    ) -> np.ndarray:
        """log p(y_i | do(T = t_i), data) for paired arrays t, y."""
        log_w, slope, var = self.predictive_components(T, Y, **kw)
        var = np.maximum(var, 1e-12)
        mu = slope[None] * t[:, None]
        lp = -0.5 * (
            np.log(2 * np.pi * var)[None] + (y[:, None] - mu) ** 2 / var[None]
        )
        return logsumexp(lp + log_w[None], axis=1)


# --------------------------------------------------------------------------- #
# Known-truth SCMs
# --------------------------------------------------------------------------- #
@dataclass
class LinearSCM:
    """x_j = sum_i B[i, j] x_i + e_j, e_j ~ N(0, s2[j])."""

    B: np.ndarray
    s2: np.ndarray

    @property
    def adj(self):
        return (self.B != 0).astype(int)

    def sample(
        self, n: int, int_node: int | None, do_values: np.ndarray | None, rng
    ):
        d = len(self.s2)
        e = rng.standard_normal((n, d)) * np.sqrt(self.s2)
        x = np.zeros((n, d))
        mask = np.zeros((n, d), dtype=bool)
        order = topo(self.adj)
        for j in order:
            if int_node is not None and j == int_node:
                x[:, j] = do_values
                mask[:, j] = True
            else:
                x[:, j] = x @ self.B[:, j] + e[:, j]
        return x, mask

    def do_dist(self, T: int, Y: int):
        """Exact (slope, var) with Y | do(T = t) ~ N(slope * t, var)."""
        d = len(self.s2)
        B = self.B.copy()
        B[:, T] = 0.0
        s2 = self.s2.copy()
        s2[T] = 0.0
        A = np.linalg.inv(np.eye(d) - B.T)
        return A[Y, T], float((A[Y] ** 2) @ s2)

    def do_logpdf(self, T, Y, t, y):
        slope, var = self.do_dist(T, Y)
        return -0.5 * (np.log(2 * np.pi * var) + (y - slope * t) ** 2 / var)


def topo(adj: np.ndarray) -> list[int]:
    d = adj.shape[0]
    indeg = adj.sum(0).astype(int).tolist()
    order, stack = [], [j for j in range(d) if indeg[j] == 0]
    while stack:
        i = stack.pop()
        order.append(i)
        for j in np.where(adj[i])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(int(j))
    return order
