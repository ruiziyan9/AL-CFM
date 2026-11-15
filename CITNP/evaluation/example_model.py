"""One fully specified linear-Gaussian example model for `kshot-example`.

A single fixed causal model -- its graph, edge weights and noise variances --
together with the do-value distribution used for its interventional rows, read
from and written to a small YAML file. The loader returns the same kind of
model object `prior_scm` returns (a `LinearSCM`, with its graph and do-value
distribution), so the kshot-example evaluation reuses the kshot-prior pipeline
unchanged.

The model is either drawn from a data config's prior on a given graph
(`draw_example_model`, via `prior_scm` with a fixed graph) or written by hand.
No model (torch) code here; the generator is imported lazily.
"""

from dataclasses import dataclass, field

import numpy as np

from CITNP.evaluation.oracle import LinearSCM
from CITNP.evaluation.tasks import prior_scm, validate_fixed_graph

# Provenance tag for a model that was not drawn from a data config's prior.
HAND_WRITTEN = "written by hand"
# Provenance tag for a model drawn from a data config's prior on a fixed graph.
FROM_PRIOR = "data config prior"


@dataclass
class ExampleModel:
    """A fixed linear-Gaussian model and the do-value distribution of its
    interventional rows.

    Attributes:
    ----------
    scm : LinearSCM
        x_j = sum_i B[i, j] x_i + e_j, e_j ~ N(0, noise_variance[j]).

    graph : np.ndarray
        [D, D] int adjacency, graph[i, j] = 1 iff i -> j (the edges with a
        nonzero weight).

    do_dist : dict
        Do-value distribution of the interventional rows, e.g.
        {"kind": "normal", "scale": 1.5} (see `sample_do_values`).

    provenance : dict
        Where the model came from: {"source": HAND_WRITTEN}, or {"source":
        FROM_PRIOR, "data_config": ..., "data_config_fingerprint": ...,
        "base_seed": ..., "task": ...} for one drawn from a data config.
    """

    scm: LinearSCM
    graph: np.ndarray
    do_dist: dict
    provenance: dict = field(
        default_factory=lambda: {"source": HAND_WRITTEN}
    )

    @property
    def num_variables(self) -> int:
        """Number of variables D."""
        return len(self.scm.s2)

    def true_effect(self, T: int, Y: int) -> float:
        """True total effect of X_T on X_Y: the slope of the true do-distribution
        X_Y | do(X_T = t) ~ N(slope t, var)."""
        return float(self.scm.do_dist(T, Y)[0])


# --------------------------------------------------------------------------- #
# Validation and (de)serialisation
# --------------------------------------------------------------------------- #
def _validated_do_dist(do_dist) -> dict:
    """Return `do_dist` as a plain dict, raising unless it is a do-value
    distribution `sample_do_values` accepts (normal with a positive scale, or
    uniform with a positive width)."""
    if not isinstance(do_dist, dict) or "kind" not in do_dist:
        raise ValueError("do_dist must be a dict with a 'kind' key")
    kind = do_dist["kind"]
    if kind == "normal":
        scale = float(do_dist["scale"])
        if not (scale > 0):
            raise ValueError("do_dist scale must be positive")
        return {"kind": "normal", "scale": scale}
    if kind == "uniform":
        width = float(do_dist["width"])
        if not (width > 0):
            raise ValueError("do_dist width must be positive")
        return {"kind": "uniform", "width": width}
    raise ValueError(f"unknown do_dist kind {kind!r}")


def _scm_from_edges(num_variables, edges, noise_variances):
    """Build and validate the `LinearSCM` and its adjacency from an edge list.

    Args:
    ----------
    num_variables : int
        Number of variables D.

    edges : list
        Each edge is (source, target, weight) or a dict with those keys;
        source -> target with coefficient weight.

    noise_variances : sequence
        The D noise variances, all positive.

    Raises ValueError unless D is a positive integer, the noise variances are
    D finite positive numbers, every edge uses distinct in-range 0-based node
    indices with a finite weight, no edge is repeated, and the graph is acyclic.
    """
    from CITNP.datasets.mixed_regime_generator import topological_order

    D = int(num_variables)
    if D <= 0:
        raise ValueError(f"num_variables must be positive, got {D}")
    s2 = np.asarray(noise_variances, dtype=float)
    if s2.shape != (D,):
        raise ValueError(
            f"noise_variances must have {D} entries, got shape {s2.shape}"
        )
    if not np.all(np.isfinite(s2)):
        raise ValueError("noise variances must be finite")
    if np.any(s2 <= 0):
        raise ValueError("noise variances must be positive")

    B = np.zeros((D, D))
    adjacency = np.zeros((D, D), dtype=int)
    for edge in edges:
        if isinstance(edge, dict):
            source, target, weight = (
                edge["source"], edge["target"], edge["weight"]
            )
        else:
            source, target, weight = edge
        source, target = int(source), int(target)
        if not (0 <= source < D and 0 <= target < D):
            raise ValueError(
                f"edge ({source}->{target}) uses a node index outside "
                f"0..{D - 1}"
            )
        if source == target:
            raise ValueError(f"edge ({source}->{target}) is a self loop")
        if adjacency[source, target]:
            raise ValueError(f"edge ({source}->{target}) is repeated")
        if not np.isfinite(weight):
            raise ValueError(f"edge ({source}->{target}) has a non-finite weight")
        B[source, target] = float(weight)
        adjacency[source, target] = 1
    try:
        topological_order(adjacency)
    except ValueError as error:
        raise ValueError(f"the example graph has a cycle ({error})") from error
    return LinearSCM(B, s2), adjacency


def example_model_to_dict(model: ExampleModel) -> dict:
    """Serialise an `ExampleModel` to plain Python types for YAML or a meta
    file. Edges are the nonzero entries of the graph, sorted by source and then
    target, each with its weight."""
    sources, targets = np.nonzero(model.graph)
    edges = [
        {
            "source": int(i),
            "target": int(j),
            "weight": float(model.scm.B[i, j]),
        }
        for i, j in zip(sources, targets)
    ]
    return {
        "num_variables": int(model.num_variables),
        "edges": edges,
        "noise_variances": [float(v) for v in model.scm.s2],
        "do_dist": _validated_do_dist(model.do_dist),
        "provenance": dict(model.provenance),
    }


def example_model_from_dict(spec: dict) -> ExampleModel:
    """Build and validate an `ExampleModel` from a dict as written by
    `example_model_to_dict` (raises ValueError on an invalid specification)."""
    if not isinstance(spec, dict):
        raise ValueError("an example model must be a mapping")
    for key in ("num_variables", "edges", "noise_variances", "do_dist"):
        if key not in spec:
            raise ValueError(f"the example model is missing '{key}'")
    scm, graph = _scm_from_edges(
        spec["num_variables"], spec["edges"], spec["noise_variances"]
    )
    do_dist = _validated_do_dist(spec["do_dist"])
    provenance = dict(spec.get("provenance", {"source": HAND_WRITTEN}))
    return ExampleModel(scm, graph, do_dist, provenance)


def write_example_model(path, model: ExampleModel) -> None:
    """Write an `ExampleModel` to `path` as YAML."""
    import yaml  # type: ignore[import-untyped]

    with open(path, "w") as f:
        yaml.safe_dump(example_model_to_dict(model), f, sort_keys=False)


def load_example_model(path) -> ExampleModel:
    """Read and validate an `ExampleModel` from a YAML file at `path`."""
    import yaml  # type: ignore[import-untyped]

    with open(path) as f:
        spec = yaml.safe_load(f)
    return example_model_from_dict(spec)


# --------------------------------------------------------------------------- #
# Drawing a model from a data config's prior
# --------------------------------------------------------------------------- #
def draw_example_model(
    cfg, graph: np.ndarray, base_seed: int, task: int = 0,
    data_config_path: str | None = None,
) -> ExampleModel:
    """Draw one model from `cfg`'s prior on the fixed `graph`, with weights,
    noise variances and the do-value distribution drawn exactly as in training
    (via `prior_scm` with a fixed graph). The graph is kept in the given
    variable order (no permutation).

    Args:
    ----------
    cfg : MixedRegimeConfig
        Data config (linear-Gaussian family, normalise: false).

    graph : np.ndarray
        [D, D] 0/1 adjacency, graph[i, j] = 1 meaning i -> j. D must be one of
        `cfg.num_variables`.

    base_seed, task : int
        Which task of the prior the model is (its weights depend only on these,
        the graph and the config).

    data_config_path : str | None
        Recorded in the model's provenance (with the config's fingerprint).
    """
    graph = validate_fixed_graph(graph, graph.shape[0], cfg)
    D = graph.shape[0]
    scm, returned_graph, do_dist, _ = prior_scm(
        cfg, task, base_seed, D, fixed_graph=graph
    )
    fingerprint = None
    if data_config_path is not None:
        from CITNP.utils.config_files import config_fingerprint

        fingerprint = config_fingerprint(data_config_path)
    provenance = {
        "source": FROM_PRIOR,
        "data_config": data_config_path,
        "data_config_fingerprint": fingerprint,
        "base_seed": int(base_seed),
        "task": int(task),
    }
    return ExampleModel(scm, returned_graph, do_dist, provenance)
