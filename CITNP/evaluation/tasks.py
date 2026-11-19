"""What to evaluate on: known-truth SCMs, context designs, k-shot experiment
pools, the training generator's graph prior and the fixed oracle bank.

No model code here (and no torch at import: the data generator is imported
lazily). Functions that draw random numbers take the generator
explicitly and draw in a fixed order, so results are reproducible.
"""

import itertools
import random
import re
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


def node_pools(scm: LinearSCM, k_max: int, do_dist: dict, rng) -> dict:
    """Interventional pools, nested in k: one per node j (every row is
    do(X_j = v)) and a "random" pool (each row targets a uniformly random
    node). Draw order: nodes 0..D-1 in order, then the random pool.

    Args:
    ----------
    scm : LinearSCM
        Ground-truth SCM.

    k_max : int
        Rows per pool.

    do_dist : dict
        Do-value distribution (see `sample_do_values`).

    rng : np.random.Generator
        Source of randomness.

    Returns:
    ----------
    {0..D-1 and "random": (x [k_max, D], int_mask [k_max, D], target node
    per row)}.
    """
    from CITNP.datasets.mixed_regime_generator import sample_do_values

    pools: dict = {}
    for node in range(len(scm.s2)):
        dv = sample_do_values(do_dist, k_max, rng)
        x, m = scm.sample(k_max, node, dv, rng)
        pools[node] = (x, m, np.full(k_max, node))
    seq = rng.integers(0, len(scm.s2), k_max)
    dv = sample_do_values(do_dist, k_max, rng)
    rx: list[np.ndarray] = []
    rm: list[np.ndarray] = []
    for i in range(k_max):
        xi, mi = scm.sample(1, int(seq[i]), dv[i : i + 1], rng)
        rx.append(xi)
        rm.append(mi)
    pools["random"] = (np.concatenate(rx), np.concatenate(rm), seq)
    return pools


def observational_pool(scm: LinearSCM, k_max: int, rng) -> tuple:
    """A pool of k_max purely OBSERVATIONAL rows from `scm`, nested in k like
    the interventional pools of `node_pools`: the first k rows are the k-row
    context. The intervention mask is all False and no node is a target (the
    per-row target array is all -1), so a context built from this pool has no
    intervened nodes. Drawn from `rng` alone, so it does not disturb the draw
    order of any other pool.

    Returns:
    ----------
    (x [k_max, D], int_mask [k_max, D] all False, target node per row all -1).
    """
    x, m = scm.sample(k_max, None, None, rng)
    return x, m, np.full(k_max, -1)


def make_pools(scm: LinearSCM, k_max: int, do_scale: float, rng) -> dict:
    """K-shot experiment pools, nested in k. For each design in DESIGNS:
    (x [k_max, D], int_mask [k_max, D], target node per row). Thin wrapper
    over `node_pools` with N(0, do_scale^2) do-values; same RNG draws as the
    original implementation (`rng.normal(0, s, n)` == `sample_do_values`)."""
    pools = node_pools(scm, k_max, {"kind": "normal", "scale": do_scale}, rng)
    out = {design: pools[node] for design, node in FIXED_NODE.items()}
    out["random"] = pools["random"]
    return out


# --------------------------------------------------------------------------- #
# Tasks drawn from the training prior (kshot-prior)
# --------------------------------------------------------------------------- #
PRIOR_DESIGNS = ["random", "do(T)", "do(Y)", "do(other)"]
# One more design, optional (see `run_kshot_prior`): extra OBSERVATIONAL rows
# from the same model, nested like the interventional pools. It shows how
# much of the gain at k rows is from more data rather than from intervention.
OBSERVATIONAL_DESIGN = "observational"


def prior_design_names(with_observational: bool = False) -> list[str]:
    """The kshot-prior design names: the four interventional designs and, when
    `with_observational`, the extra-observational-rows design last."""
    return PRIOR_DESIGNS + ([OBSERVATIONAL_DESIGN] if with_observational else [])


def example_design_name(pool_key) -> str:
    """Design label for `kshot-example`, named by what the pool adds to the
    context rather than by a role: "do(X{j+1})" for a pool intervening on node
    j (node j is variable X_{j+1}), and "random" or "observational" for those
    pools."""
    if pool_key in ("random", OBSERVATIONAL_DESIGN):
        return str(pool_key)
    return f"do(X{int(pool_key) + 1})"


def example_design_names(D: int) -> list[str]:
    """Every `kshot-example` design, in a fixed order: do(X1) .. do(XD), then
    the random pool and the extra-observational pool."""
    return (
        [example_design_name(j) for j in range(D)]
        + ["random", OBSERVATIONAL_DESIGN]
    )


def validate_fixed_graph(graph, D: int, cfg) -> np.ndarray:
    """Check a user-given DAG for the fixed-graph evaluation and return it as
    a [D, D] int array (graph[i, j] = 1 iff i -> j).

    Raises ValueError unless `graph` is a square 0/1 matrix of side D,
    without self loops and acyclic, and D is one of `cfg.num_variables`.
    """
    from CITNP.datasets.mixed_regime_generator import topological_order

    adj = np.asarray(graph)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"fixed graph must be square, got shape {adj.shape}")
    if adj.shape[0] != D:
        raise ValueError(
            f"fixed graph has {adj.shape[0]} nodes but D = {D}"
        )
    if D not in [int(d) for d in cfg.num_variables]:
        raise ValueError(
            f"D = {D} is not in the data config's num_variables "
            f"{list(cfg.num_variables)}"
        )
    if not np.isin(adj, (0, 1)).all():
        raise ValueError("fixed graph must contain only 0 and 1")
    if np.diag(adj).any():
        raise ValueError("fixed graph has a self loop")
    adj = adj.astype(int)
    try:
        topological_order(adj)
    except ValueError as error:
        raise ValueError(f"fixed graph has a cycle ({error})") from error
    return adj


def format_graph_edges(graph) -> str:
    """Edge list of a DAG as text, "0->1,0->2,1->2" (graph[i, j] = 1 gives
    "i->j"), sorted by source and then target; "" for a graph without edges."""
    sources, targets = np.nonzero(np.asarray(graph))
    return ",".join(f"{int(i)}->{int(j)}" for i, j in zip(sources, targets))


def parse_graph_edges(text: str, D: int) -> np.ndarray:
    """Adjacency matrix from an edge list such as "0->1, 1->2, 0->2": 0-based
    node indices, "i->j" meaning i -> j, edges separated by commas (spaces
    are ignored).

    Returns:
    ----------
    [D, D] int array, adjacency[i, j] = 1 iff the list has "i->j".

    Raises ValueError for an empty list, an entry that is not "i->j" with
    non-negative integers, an index >= D, a repeated edge, a self loop and a
    cycle.
    """
    entries = [e.strip() for e in text.replace(" ", "").split(",")]
    if text.strip() == "":
        raise ValueError("the edge list is empty")
    adjacency = np.zeros((D, D), dtype=int)
    for entry in entries:
        match = re.fullmatch(r"(\d+)->(\d+)", entry)
        if match is None:
            raise ValueError(
                f"bad edge {entry!r}: expected 'source->target' with "
                "non-negative integer node indices, e.g. '0->1,1->2'"
            )
        source, target = int(match.group(1)), int(match.group(2))
        if max(source, target) >= D:
            raise ValueError(
                f"edge {entry!r} uses a node index >= D = {D} (indices are "
                "0-based)"
            )
        if source == target:
            raise ValueError(f"edge {entry!r} is a self loop")
        if adjacency[source, target]:
            raise ValueError(f"edge {entry!r} is repeated")
        adjacency[source, target] = 1
    from CITNP.datasets.mixed_regime_generator import topological_order

    try:
        topological_order(adjacency)
    except ValueError as error:
        raise ValueError(f"the edges form a cycle ({error})") from error
    return adjacency


def prior_scm(
    cfg, i: int, base_seed: int = 9101, D: int = 3, fixed_graph=None
):
    """SCM, graph and do-value distribution of task `i` of the training prior,
    optionally with the graph fixed.

    Calls the training generator with key make_key(SOURCE_ADHOC, D,
    base_seed, i) and keeps only the SCM, graph and do_dist; the generator's
    own context and query rows are discarded. Depends only on (cfg,
    base_seed, D, i): D sits in the key's split slot, so tasks with the same
    i but different D use unrelated random streams. Other SOURCE_ADHOC
    callers use split 0 and D >= 1, so these keys collide with none of them.

    With `fixed_graph` the graph is not sampled and the variables are not
    permuted: every task has that DAG in that variable order. Mechanisms
    (weights, noise variances) and the do-value distribution are drawn by the
    same code and the same prior as for a sampled graph, so only the graph
    differs from the prior's tasks. The random stream differs from the
    prior's for the same (base_seed, D, i), since no graph is drawn.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Data config (linear-Gaussian family, normalise: false).

    i : int
        Task index.

    base_seed : int
        Seed of the task stream (distinct from the bank and calibrate seeds).

    D : int
        Number of variables.

    fixed_graph : np.ndarray | None
        [D, D] 0/1 adjacency, fixed_graph[i, j] = 1 meaning i -> j. None
        draws each task's graph from the training prior, with the prior's
        random variable permutation (the default).

    Returns:
    ----------
    (LinearSCM, graph [D, D] int in the task's variable order, do_dist dict,
    number of resample attempts the generator needed).
    """
    from CITNP.datasets.mixed_regime_generator import (
        SOURCE_ADHOC,
        generate_task,
        make_key,
    )
    from CITNP.prior_diagnostics.linear_gaussian import scm_from_debug

    require_exact_oracle(cfg)
    if fixed_graph is not None:
        fixed_graph = validate_fixed_graph(fixed_graph, D, cfg)
    task, debug = generate_task(
        cfg,
        make_key(SOURCE_ADHOC, D, base_seed, i),
        graph=fixed_graph,
        num_variables=D,
        return_debug=True,
        permute=fixed_graph is None,
    )
    lg = scm_from_debug(debug)
    if not np.allclose(lg.c, 0.0):
        raise ValueError("prior_scm needs zero intercepts (normalise: false)")
    return (
        LinearSCM(lg.B, lg.omega),
        task.graph.astype(int),
        dict(task.meta["do_dist"]),
        int(task.meta["resample_attempts"]),
    )


def pair_design_pools(
    T: int, Y: int, D: int, with_observational: bool = False
) -> dict:
    """Pool(s) used by each design for the query pair (T, Y): "random" -> the
    random pool, "do(T)" -> pool T, "do(Y)" -> pool Y, "do(other)" -> the list
    of the remaining D - 2 nodes' pools. With `with_observational`,
    "observational" -> the extra-observational pool, the same for every pair."""
    pools: dict = {
        "random": "random",
        "do(T)": T,
        "do(Y)": Y,
        "do(other)": [j for j in range(D) if j not in (T, Y)],
    }
    if with_observational:
        pools[OBSERVATIONAL_DESIGN] = OBSERVATIONAL_DESIGN
    return pools


def default_n_obs(cfg) -> int:
    """Smallest observational context the training prior can produce:
    max(n_ctx_range[0], n_obs_min)."""
    return max(int(cfg.n_ctx_range[0]), int(getattr(cfg, "n_obs_min", 0)))


def check_context_size(cfg, n_obs: int, k_max: int) -> None:
    """Raise if n_obs + k_max rows exceed the training n_ctx_range."""
    if n_obs + k_max > cfg.n_ctx_range[1]:
        raise ValueError(
            f"n_obs + k_max = {n_obs + k_max} exceeds the training "
            f"n_ctx_range {tuple(cfg.n_ctx_range)}"
        )


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


def resolve_graph_prior_samples(D: int, n: int | None = None) -> int:
    """Graphs sampled by `empirical_graph_prior` for the kshot-prior oracle:
    `n` if given, else 20000 for D <= 3 (all 25 DAGs are seen) and 500000
    for D >= 4 (543 DAGs; 20000 can miss ~1% of the prior mass, and the
    oracle gives unseen DAGs log prior -inf)."""
    if n is not None:
        return int(n)
    return 20000 if D <= 3 else 500000


# --------------------------------------------------------------------------- #
# Fixed oracle bank
# --------------------------------------------------------------------------- #
BUCKETS = ("nT=0", "nT=1-10", "nT=11+")


def n_int_bucket(n: int) -> str:
    """Bucket of the number of context rows intervened on the query treatment."""
    if n == 0:
        return BUCKETS[0]
    return BUCKETS[1] if n <= 10 else BUCKETS[2]


def require_exact_oracle(cfg) -> None:
    """Raise unless the exact oracle applies to `cfg`: a linear-Gaussian
    family with normalise: false."""
    if not is_linear_gaussian(cfg.function_generator) or cfg.normalise:
        raise ValueError(
            "oracle bank needs a linear-Gaussian family with normalise: false"
        )


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

    require_exact_oracle(cfg)
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
