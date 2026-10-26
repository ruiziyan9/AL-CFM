"""What to evaluate on: known-truth SCMs, context designs, k-shot experiment
pools, the training generator's graph prior and the fixed oracle bank.

No model code here (and no torch at import: the data generator is imported
lazily). Functions that draw random numbers take the generator
explicitly and draw in a fixed order, so results are reproducible.
"""

import itertools
import random
import time

import numpy as np

from CITNP.datasets.linear_gaussian_prior import (
    LinearGaussianPrior,
    is_linear_gaussian,
)
from CITNP.evaluation.oracle import LinearSCM, LinGaussOracle, dag_key

# --------------------------------------------------------------------------- #
# Confounder SCM roles and the six marginal interventional queries
# --------------------------------------------------------------------------- #
TREAT, CONF, OUTC = 0, 1, 2
ROLE = {TREAT: "treatment", CONF: "confounder", OUTC: "outcome"}
QUERIES = [
    (T, Y) for T in range(3) for Y in range(3) if T != Y
]  # (treatment, outcome)
MAIN_QUERY = f"{ROLE[OUTC]}|do({ROLE[TREAT]})"
DESIGNS = ["random", "do(treatment)", "do(confounder)", "do(outcome)"]
FIXED_NODE = {"do(treatment)": TREAT, "do(confounder)": CONF, "do(outcome)": OUTC}


def qname(T: int, Y: int) -> str:
    """Query label from the node roles, e.g. 'outcome|do(treatment)'."""
    return f"{ROLE[Y]}|do({ROLE[T]})"


# Identifiability labels (confounder graph, true order X1, X0, X2)
_ALL = frozenset(QUERIES)


def identified_queries(nodes: frozenset) -> frozenset:
    """Queries whose effect is identified, as k grows, by interventions on
    `nodes` (plus observational data), for the confounder graph."""
    if TREAT in nodes or {CONF, OUTC} <= nodes:
        return _ALL  # the causal order X1, X0, X2 is pinned down
    if nodes == {CONF}:
        # X1 first; X0 - X2 orientation open
        return frozenset(
            {(CONF, TREAT), (CONF, OUTC), (TREAT, CONF), (OUTC, CONF)}
        )
    if nodes == {OUTC}:
        # X2 last; X0 - X1 orientation open
        return frozenset({(OUTC, TREAT), (OUTC, CONF)})
    return frozenset()


def query_type(nodes: frozenset, T: int, Y: int) -> str:
    """'seen', 'unseen_identified' or 'unidentified' given intervened nodes."""
    if T in nodes:
        return "seen"
    if (T, Y) in identified_queries(nodes):
        return "unseen_identified"
    return "unidentified"


# --------------------------------------------------------------------------- #
# Known-truth SCMs
# --------------------------------------------------------------------------- #
def standardise(scm: LinearSCM) -> LinearSCM:
    """Rescale every variable to unit marginal (observational) variance.
    x'_j = x_j / sd_j  =>  B'[i, j] = B[i, j] sd_i / sd_j,  s2'_j = s2_j / sd_j^2."""
    A = np.linalg.inv(np.eye(len(scm.s2)) - scm.B.T)
    sd = np.sqrt(np.diag(A @ np.diag(scm.s2) @ A.T))
    return LinearSCM(scm.B * sd[:, None] / sd[None, :], scm.s2 / sd**2)


def confounder_scm(
    b_ct: float = 0.9, b_cy: float = 0.8, b_ty: float = -0.3, s2=(0.2, 0.3, 0.25)
) -> LinearSCM:
    """Standardised confounder SCM X1 -> X0, X1 -> X2, X0 -> X2."""
    B = np.zeros((3, 3))
    B[CONF, TREAT], B[CONF, OUTC], B[TREAT, OUTC] = b_ct, b_cy, b_ty
    return standardise(LinearSCM(B, np.asarray(s2, dtype=float)))


def random_confounder_scm(rng: np.random.Generator) -> LinearSCM:
    """Confounder SCM with random signs, weights and noise variances."""

    def sign() -> float:
        return rng.choice([-1.0, 1.0])

    return confounder_scm(
        b_ct=sign() * rng.uniform(0.5, 1.5),
        b_cy=sign() * rng.uniform(0.5, 1.5),
        b_ty=sign() * rng.uniform(0.2, 1.0),
        s2=rng.uniform(0.1, 1.0, 3),
    )


def scm_suite() -> list[tuple[str, LinearSCM, tuple[int, int], str]]:
    """Fixed 3-variable SCMs: (name, SCM, query (T, Y), why it is interesting)."""
    D = 3
    out = []

    B = np.zeros((D, D))
    B[0, 1], B[1, 2] = 0.9, -0.8
    out.append((
        "chain_0to1to2", LinearSCM(B, np.array([0.3, 0.2, 0.25])), (0, 2),
        "obs MEC = {chain, reverse chain, fork}: effect not identified",
    ))  # fmt: skip

    B = np.zeros((D, D))
    B[1, 0], B[1, 2] = 0.9, 0.8
    out.append((
        "fork_1_to_0_2", LinearSCM(B, np.array([0.2, 0.3, 0.25])), (0, 2),
        "same MEC, true effect is ZERO (confounded association)",
    ))  # fmt: skip

    B = np.zeros((D, D))
    B[0, 2], B[1, 2] = 0.9, 0.7
    out.append((
        "collider_0_1_to_2", LinearSCM(B, np.array([0.3, 0.3, 0.2])), (0, 2),
        "v-structure: identified from obs, experiments add little",
    ))  # fmt: skip
    return out


# --------------------------------------------------------------------------- #
# Contexts
# --------------------------------------------------------------------------- #
def context_designs(
    n_obs: int = 128, m_grid=(0, 8, 16, 32, 64, 128), num_nodes: int = 3
) -> list[tuple[str, list[tuple[int | None, int]]]]:
    """(label, regimes). Total rows stay inside training n_ctx_range; every
    design keeps n_obs observational rows (normalisation needs them)."""
    out: list[tuple[str, list[tuple[int | None, int]]]] = [
        ("obs_only", [(None, n_obs)])
    ]
    for node, m in itertools.product(range(num_nodes), m_grid[1:]):
        out.append((f"obs+do(X{node})x{m}", [(None, n_obs), (node, m)]))
    return out


def make_context(
    scm: LinearSCM, regimes: list[tuple[int | None, int]], do_scale: float, rng
) -> tuple[np.ndarray, np.ndarray]:
    """regimes: list of (intervened node or None for obs, n_rows). Returns x, mask."""
    xs: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    for node, n in regimes:
        if n == 0:
            continue
        dv = rng.normal(0, do_scale, n) if node is not None else None
        x, m = scm.sample(n, node, dv, rng)
        xs.append(x)
        ms.append(m)
    x, m = np.concatenate(xs), np.concatenate(ms)
    perm = rng.permutation(len(x))
    return x[perm], m[perm]


def make_pools(scm: LinearSCM, k_max: int, do_scale: float, rng) -> dict:
    """K-shot experiment pools, nested in k. For each design in DESIGNS:
    (x [k_max, D], int_mask [k_max, D], target node per row)."""
    pools = {}
    for design, node in FIXED_NODE.items():
        dv = rng.normal(0, do_scale, k_max)
        x, m = scm.sample(k_max, node, dv, rng)
        pools[design] = (x, m, np.full(k_max, node))
    seq = rng.integers(0, len(scm.s2), k_max)
    dv = rng.normal(0, do_scale, k_max)
    rx: list[np.ndarray] = []
    rm: list[np.ndarray] = []
    for i in range(k_max):
        xi, mi = scm.sample(1, int(seq[i]), dv[i : i + 1], rng)
        rx.append(xi)
        rm.append(mi)
    pools["random"] = (np.concatenate(rx), np.concatenate(rm), seq)
    return pools


# --------------------------------------------------------------------------- #
# Graph prior of the training generator (incl. its random variable permutation)
# --------------------------------------------------------------------------- #
def motif_graph_prior(cfg) -> dict:
    """dag_key -> exact log prior of a MOTIF3 config: log(class probability /
    class size) for each of the 25 labelled DAGs, -inf for classes with
    probability 0."""
    from CITNP.datasets.motifs import MOTIF_SIZES, motif_class
    from CITNP.evaluation.oracle import all_dags

    out = {}
    for g in all_dags(3):
        c = motif_class(g)
        p = cfg.motif_probs[c] / MOTIF_SIZES[c]
        out[dag_key(g)] = float(np.log(p)) if p > 0 else -np.inf
    return out


def empirical_graph_prior(
    cfg, n: int = 20000, seed: int = 0, num_nodes: int = 3
) -> dict:
    """dag_key -> log frequency over n graphs from the training generator.
    Reseeds the global numpy and `random` RNGs (sample_graph uses them).
    MOTIF3 configs return the exact `motif_graph_prior` instead (no sampling)."""
    if list(cfg.graph_type) == ["MOTIF3"]:
        return motif_graph_prior(cfg)
    from CITNP.datasets.mixed_regime_generator import sample_graph

    np.random.seed(seed)
    random.seed(seed)
    counts: dict = {}
    for _ in range(n):
        g, _ = sample_graph(cfg, num_nodes)
        p = np.random.permutation(num_nodes)
        g = g[np.ix_(p, p)].astype(int)
        counts[dag_key(g)] = counts.get(dag_key(g), 0) + 1
    return {k: np.log(v / n) for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Fixed oracle bank
# --------------------------------------------------------------------------- #
BUCKETS = ("nT=0", "nT=1-10", "nT=11+")


def n_int_bucket(n: int) -> str:
    """Bucket of the number of context rows intervened on the query treatment."""
    if n == 0:
        return BUCKETS[0]
    return BUCKETS[1] if n <= 10 else BUCKETS[2]


def build_bank(
    cfg, per_bucket=150, D=3, base_seed=4242, n_mc=2000, max_tasks=100_000
) -> list[dict]:
    """Return a list of question dicts; `per_bucket` questions (T) per bucket,
    each expanded over all D - 1 outcomes Y != T."""
    from CITNP.datasets.mixed_regime_generator import (
        SOURCE_ADHOC,
        generate_task,
        make_key,
    )

    if not is_linear_gaussian(cfg.function_generator) or cfg.normalise:
        raise ValueError(
            "oracle bank needs a linear-Gaussian family with normalise: false"
        )
    lg_prior = LinearGaussianPrior.from_config(cfg)
    prior = empirical_graph_prior(cfg, num_nodes=D)
    counts = {b: 0 for b in BUCKETS}
    items = []
    t0 = time.time()
    for i in range(max_tasks):
        if all(c >= per_bucket for c in counts.values()):
            break
        # SOURCE_ADHOC keys never collide with training / val / test streams.
        task = generate_task(
            cfg, make_key(SOURCE_ADHOC, 0, base_seed, i), num_variables=D
        )
        ctx_x = task.ctx_x.astype(np.float64)
        ctx_int = task.ctx_int.astype(bool)
        qx = task.qry_x.astype(np.float64)
        qi = task.qry_int.astype(bool)
        # Grouped mode: each query row intervenes on exactly one variable.
        treatments = sorted({int(np.argmax(r)) for r in qi})

        todo = []
        for T in treatments:
            n_T = int(ctx_int[:, T].sum())
            b = n_int_bucket(n_T)
            if counts[b] < per_bucket:
                todo.append((T, b, n_T))
        if not todo:
            continue

        orc = LinGaussOracle(
            D,
            log_graph_prior=prior,
            prior=lg_prior,
        ).fit(ctx_x, ctx_int)
        K = int(task.meta["K"])
        has_obs = bool(task.meta["has_obs"])
        for T, b, n_T in todo:
            rows = qi[:, T]
            t = qx[rows, T]
            for Y in range(D):
                if Y == T:
                    continue
                y = qx[rows, Y]
                items.append(
                    dict(
                        ctx_x=task.ctx_x.astype(np.float32),
                        ctx_int=ctx_int,
                        T=T,
                        Y=Y,
                        t=t.astype(np.float32),
                        y=y.astype(np.float32),
                        oracle_logp=orc.logpdf(T, Y, t, y, n_mc=n_mc, seed=i),
                        n_int_T=n_T,
                        labels=[
                            "all",
                            b,
                            f"K={'2+' if K >= 2 else K}",
                            f"has_obs={has_obs}",
                        ],
                    )
                )
            counts[b] += 1
        if i % 200 == 0:
            print(f"task {i}: {counts}  ({time.time() - t0:.0f}s)")
    else:
        print(
            f"WARNING: hit max_tasks={max_tasks} before filling quotas: {counts}"
        )
    return items
