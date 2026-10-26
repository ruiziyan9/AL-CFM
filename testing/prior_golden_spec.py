"""Shared specification of the prior goldens (generator and oracle).

Used by `testing/make_prior_golden.py` (records the golden outputs with the
trusted pre-refactor code) and `testing/test_prior_golden.py` (reruns them
with the current code), so both make exactly the same calls. Later refactor
steps only need to update the import paths inside the functions below.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "testing" / "golden" / "prior"

N_TASKS = 20
D = 3
CONFIGS = REPO / "CITNP" / "experiments" / "configs"

# Same settings as BASE in testing/test_signed_mean_prior.py, weight_mean = 1.
SIGNED_MEAN = dict(
    function_generator="linearnonidentifiable",
    num_variables=[3],
    graph_type=["ER"],
    graph_degrees={3: [1, 2, 3]},
    normalise=False,
    n_ctx_range=(64, 128),
    n_qry=48,
    qry_mode="grouped",
    weight_mean=1.0,
)
# name -> yaml file name, or a dict of MixedRegimeConfig kwargs
GENERATOR_SETTINGS: dict[str, str | dict] = {
    "oracle_d3": "oracle_d3.yaml",
    "oracle_d3_strong_v9": "oracle_d3_strong_v9.yaml",
    "signed_mean": SIGNED_MEAN,
}
TASK_ARRAYS = [
    "ctx_x",
    "ctx_int",
    "ctx_regime",
    "qry_x",
    "qry_int",
    "qry_tgt",
    "graph",
    "qry_group",
]

# (weight_scale eta, weight_mean m) settings of the oracle golden
ORACLE_SETTINGS = [(1.0, 0.0), (9.0, 0.0), (1.0, 0.5)]
ORACLE_N = 100


def generator_config(name: str):
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.datasets.mixed_regime_generator import MixedRegimeConfig

    spec = GENERATOR_SETTINGS[name]
    if isinstance(spec, str):
        return load_config(CONFIGS / spec)
    return MixedRegimeConfig(**spec)


def generator_arrays(name: str) -> dict[str, np.ndarray]:
    """Every array field of N_TASKS tasks at fixed keys, as `t{i}_{field}`.
    regime_targets is stored as a JSON string."""
    from CITNP.datasets.mixed_regime_generator import (
        SOURCE_ADHOC,
        generate_task,
        make_key,
    )

    cfg = generator_config(name)
    out: dict[str, np.ndarray] = {}
    for i in range(N_TASKS):
        key = make_key(SOURCE_ADHOC, 0, 0, i)
        task = generate_task(cfg, key, num_variables=D)
        for f in TASK_ARRAYS:
            out[f"t{i}_{f}"] = np.asarray(getattr(task, f))
        out[f"t{i}_regime_targets"] = np.array(
            json.dumps([list(t) for t in task.regime_targets])
        )
    return out


def oracle_datasets() -> list[tuple[np.ndarray, np.ndarray]]:
    """Three synthetic (x [n, 3], int_mask [n, 3]) datasets; the last has
    interventional rows."""
    n = ORACLE_N
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(n, D))  # chain-like
    x0[:, 1] += 1.5 * x0[:, 0]
    x0[:, 2] += -0.8 * x0[:, 0] + 1.2 * x0[:, 1]

    rng = np.random.default_rng(1)
    x1 = rng.normal(size=(n, D))  # collider-like
    x1[:, 2] = x1[:, 0] - x1[:, 1] + 0.5 * rng.normal(size=n)

    rng = np.random.default_rng(2)
    x2 = rng.normal(size=(n, D))
    x2[:, 1] += 1.0 * x2[:, 0]
    x2[:, 2] += 0.7 * x2[:, 1]
    m2 = np.zeros((n, D), dtype=bool)
    for r in range(70, n):  # last 30 rows: one do-intervention each
        t = int(rng.integers(D))
        x2[r, t] = rng.normal(0.0, 2.0)
        m2[r, t] = True

    z = np.zeros((n, D), dtype=bool)
    return [(x0, z), (x1, z.copy()), (x2, m2)]


def oracle_arrays() -> dict[str, np.ndarray]:
    """Total log marginal likelihood of all 25 DAGs for every dataset and
    (eta, m) setting, plus the DAG order (`dags`)."""
    from CITNP.evaluation.oracle import all_dags, node_posterior

    dags = all_dags(D)
    out: dict[str, np.ndarray] = {"dags": np.stack(dags)}
    for d, (x, mask) in enumerate(oracle_datasets()):
        for eta, m in ORACLE_SETTINGS:
            vals = []
            for g in dags:
                total = 0.0
                for j in range(D):
                    pa = np.where(g[:, j])[0]
                    total += node_posterior(
                        x, mask, j, pa, eta=eta, weight_mean=m
                    ).log_ml
                vals.append(total)
            out[f"ds{d}_eta{eta:g}_m{m:g}"] = np.array(vals)
    return out
