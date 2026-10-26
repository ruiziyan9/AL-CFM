"""
Mixed-regime task generator.

Each task is one SCM (graph + mechanisms) and one randomly sampled design. The
context mixes an (optional) observational regime with several single-node
interventional regimes from the same SCM; query rows each carry their own
(single-node or empty) intervention and all their non-intervened cells are
prediction targets.

Pipeline for one task (`generate_task`):
    1. seed `random`, `np.random` and `torch` from the task key (`make_key`);
    2. sample D, a DAG (`sample_graph`) and a design (`sample_design`);
    3. instantiate the mechanisms of the chosen function family;
    4. stack [reference | context regimes | queries] rows into one buffer and
       generate every node in topological order with one mechanism call;
    5. randomly permute the variables, drop the reference rows and shuffle the
       context rows.

Invariant: every node's mechanism is called EXACTLY ONCE per task, on all rows
(reference, context and query) stacked together. Mechanisms in
`functions_generator.py` are not fixed functions (GPs draw a fresh joint sample,
some linear families resample weights per call), so splitting the evaluation
would break the "same SCM" guarantee.

Example:
    cfg = MixedRegimeConfig(
        function_generator="linearnonidentifiable",
        num_variables=[2, 3, 4, 5],
        graph_type=["ER"],
        graph_degrees={2: [0, 1], 3: [1, 2, 3], 4: [1, 2, 4], 5: [2, 4, 6]},
    )
    task = generate_task(cfg, seed=0)               # one unpadded Task
    batch = collate_tasks([generate_task(cfg, s) for s in range(8)])

See MIXED_REGIME_README.md for the config fields and the batch contract.
"""

import dataclasses
import glob
import json
import math
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from CITNP.datasets.causal_graph_generator import generate_synthetic_dag
from CITNP.datasets.functions_generator import (
    GPFunctionGenerator,
    LinearFixedStdFunctionGenerator,
    LinearIWFunctionGenerator,
    LinearNIGFunctionGenerator,
    LinearNIGLegacyFunctionGenerator,
    LinearNonIdentifiableFunctionGenerator,
    NeuralGPLVMFunctionGenerator,
    NeuralNetFunctionGenerator,
    NeuralSimpleGPFunctionGenerator,
    ResNeuralFunctionGenerator,
    ResNeuralGPLVMFunctionGenerator,
    SimpleGPFunctionGenerator,
    SimpleGPLVMFunctionGenerator,
    SinusoidFunctionGenerator,
    UniformLinearFunctionGenerator,
)
from CITNP.datasets.motifs import MOTIF_CLASSES, motif_dags, validate_motif_probs
from CITNP.datasets.linear_gaussian_prior import (
    LinearGaussianPrior,
    is_legacy_key,
    is_linear_gaussian,
)

# Same registry as BaseInterventionDatasetGenerator.__init__ (kept in sync by hand).
REGISTRY: dict[str, type[Any]] = {
    "gp": GPFunctionGenerator,
    "sinusoid": SinusoidFunctionGenerator,
    "neuralnet": NeuralNetFunctionGenerator,
    "simplegp": SimpleGPFunctionGenerator,
    "simplegplvm": SimpleGPLVMFunctionGenerator,
    "uniformlinear": UniformLinearFunctionGenerator,
    "linearfixedstd": LinearFixedStdFunctionGenerator,
    "linear_iw": LinearIWFunctionGenerator,
    "linear_nig": LinearNIGFunctionGenerator,
    "linear_nig_legacy": LinearNIGLegacyFunctionGenerator,
    # Deprecated alias of linear_nig_legacy (warns once per process).
    "linearnonidentifiable": LinearNonIdentifiableFunctionGenerator,
    "neuralgplvm": NeuralGPLVMFunctionGenerator,
    "neuralsimplegp": NeuralSimpleGPFunctionGenerator,
    "resnet": ResNeuralFunctionGenerator,
    "resnetgplvm": ResNeuralGPLVMFunctionGenerator,
}

# Families whose mechanisms build an [N, N] kernel matrix (O(N^3) sampling).
GP_FAMILIES = {
    "gp",
    "simplegp",
    "simplegplvm",
    "neuralgplvm",
    "neuralsimplegp",
    "resnetgplvm",
}

# Families whose mechanisms never call th.cuda.is_available() (checked against
# functions_generator.py): they always run on the CPU.
CPU_ONLY_FAMILIES = {
    "resnet",
    "linearfixedstd",
    "linearnonidentifiable",
    "linear_nig_legacy",
    "linear_nig",
    "linear_iw",
    "uniformlinear",
    "sinusoid",
}

# Diagnostic counter of rejected task attempts by reason (see
# prior_diagnostics.resample); read-only, never changes the generated tasks.
RESAMPLE_COUNTS = {"nonfinite": 0, "too_large": 0}

# Query row types, in the order used by `MixedRegimeConfig.qry_type_probs`.
QRY_TYPES = ("seen", "unseen", "empty")

# --------------------------------------------------------------------------- #
# Seeding
#
# Every task is identified by a fixed-length key
#     (source, split_id, base_seed, i1, i2, i3)
# and all RNG states are derived from it with np.random.SeedSequence, which
# hashes the whole key. Different keys give unrelated streams, so splits,
# base seeds, epochs, workers and sources can never share tasks (unlike
# additive seed offsets, where base seeds 0 and 1 overlap almost entirely).
# --------------------------------------------------------------------------- #

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
# First element of every key: where the task comes from (domain separation).
SOURCE_ADHOC, SOURCE_CHUNK, SOURCE_STREAM, SOURCE_STREAM_D = 0, 1, 2, 3


def make_key(
    source: int,
    split_id: int,
    base_seed: int,
    i1: int = 0,
    i2: int = 0,
    i3: int = 0,
) -> tuple[int, ...]:
    """Fixed-length (6) task key. All entries must be non-negative ints.

    Conventions:
        SOURCE_ADHOC:    (0, 0, base_seed, i, 0, 0)   tests, diagnostics, int seeds
        SOURCE_CHUNK:    (1, split, base_seed, global_task_index, 0, 0)
        SOURCE_STREAM:   (2, split, base_seed, epoch, worker_id, counter)
        SOURCE_STREAM_D: (3, split, base_seed, epoch, worker_id, batch_idx)
    """
    key = (source, split_id, base_seed, i1, i2, i3)
    for k in key:
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError(
                f"Seed key entries must be non-negative ints, got {key}"
            )
    return tuple(int(k) for k in key)


def _seed_all(key: tuple[int, ...]) -> None:
    """Seed python `random` (igraph), np.random (legacy global) and torch from a key."""
    words = np.random.SeedSequence(list(key)).generate_state(6, dtype=np.uint32)
    np.random.seed(words[:4])  # 128-bit legacy seed
    torch.manual_seed((int(words[4]) << 32) | int(words[5]))  # 64-bit
    random.seed(int.from_bytes(words.tobytes(), "little"))


def assert_disjoint_splits(*chunk_dirs: str) -> None:
    """Raise if two chunk directories could contain the same tasks.

    Reads config.json in each directory. Two directories conflict if they have
    the same split_id AND the same base_seed (then their keys overlap).
    """
    seen: dict[tuple[int, int], str] = {}
    for d in chunk_dirs:
        path = os.path.join(d, "config.json")
        with open(path) as f:
            info = json.load(f)
        if "split_id" not in info:
            raise ValueError(
                f"{path} has no split_id: chunks predate key-based seeding "
                "and must be regenerated"
            )
        sig = (int(info["split_id"]), int(info["base_seed"]))
        if sig in seen:
            raise ValueError(
                f"{seen[sig]} and {d} share split_id={sig[0]} and "
                f"base_seed={sig[1]}, so they contain the same tasks"
            )
        seen[sig] = d


@dataclass
class MixedRegimeConfig:
    """Prior over SCMs and designs. All fields are listed in the README.

    Args:
    ----------
    function_generator : str
        Mechanism family, a key of `REGISTRY`.

    num_variables : list[int]
        D is sampled uniformly from this list for each task.

    graph_type : list[str]
        Graph types passed to `generate_synthetic_dag` ("ER", "SF", "BP", "FC").

    graph_degrees : list[int] | dict[int, list[int]]
        Expected number of EDGES (s0), either one list for all D or a dict
        keyed by D.

    normalise : bool
        Standardise every variable with statistics of the hidden reference
        rows. Set False for oracle-matched families.

    n_ref : int
        Number of hidden observational reference rows.

    p_obs, k_max, p_geom :
        P(context has an observational regime); max number of interventional
        regimes (capped at D); P(K=k) ∝ (1 - p_geom)^k.

    n_ctx_range, dirichlet_alpha_range :
        the range of how many context rows to generate, including obs and int regimes
        Total context rows ~ U{n_ctx_range}; split across regimes with a
        Dirichlet(alpha) whose alpha is log-uniform on the range.

    do_scale_range, p_wide_do, wide_do_range :
        Do-values are N(0, s^2) with s ~ U(do_scale_range), or with probability
        p_wide_do Uniform(-w, w) with w ~ U(wide_do_range).

    n_qry, qry_type_probs :
        Query rows per task and weights of the (seen, unseen, empty) types.

    max_total_rows_gp : int
        Upper bound on total rows for GP families.

    max_resample_attempts : int
        Retries when a task contains NaN/inf (or values above max_abs_value)
        before raising.

    max_abs_value : float
        Resample a task if any generated value (reference, context or query)
        exceeds this in absolute value.

    qry_mode : str
        "per_row" (default): every query row has its own target set (seen,
        unseen or empty), as before. "grouped": each task has G question
        groups; group g is a contiguous block of n_qry // G query rows that
        all intervene on the same single treatment T_g (distinct across
        groups). The "empty" type is not used in grouped mode (upstream's
        role embedding needs a treatment), so qry_type_probs[2] is ignored.

    qry_groups : list[int]
        Allowed numbers of groups G (grouped mode only). Each must divide n_qry
        and be <= min(num_variables).
    """

    function_generator: str
    num_variables: list[int]
    graph_type: list[str]
    # Expected number of EDGES passed to generate_synthetic_dag as s0 (same
    # convention as the original repo). Either a list or a dict keyed by D.
    graph_degrees: list[int] | dict[int, list[int]]
    normalise: bool = True
    n_ref: int = 256
    p_obs: float = 0.85
    # Minimum number of observational rows per task. When > 0, every task has
    # an observational regime (p_obs is ignored) with at least this many rows,
    # so per-task normalisation can always use observational statistics.
    n_obs_min: int = 10
    k_max: int = 4
    p_geom: float = 0.5
    n_ctx_range: tuple[int, int] = (64, 1024)
    dirichlet_alpha_range: tuple[float, float] = (0.3, 3.0)
    do_scale_range: tuple[float, float] = (0.5, 3.0)
    p_wide_do: float = 0.1
    wide_do_range: tuple[float, float] = (3.0, 6.0)
    n_qry: int = 128
    # Weights for the (seen, unseen, empty) query row types. Mostly
    # interventional: interventional prediction is the main target.
    qry_type_probs: tuple[float, float, float] = (0.45, 0.45, 0.10)
    max_total_rows_gp: int = 2048
    # Retries when a task is non-finite or exceeds max_abs_value.
    max_resample_attempts: int = 10
    # Keeps values far below the float32 max, so the final cast cannot overflow.
    max_abs_value: float = 1e4
    # Query layout: "per_row" (default, unchanged) or "grouped".
    qry_mode: str = "per_row"
    # Allowed numbers of question groups G (grouped mode only).
    qry_groups: list[int] = field(default_factory=lambda: [1, 2, 3])
    # Edge-weight scale eta of linearnonidentifiable: w | s2 ~ N(0, eta * s2).
    # 1.0 is the original prior; the exact oracle must use the same value.
    weight_scale: float = 1.0
    # Edge-weight mean m of linearnonidentifiable: w | s2 ~ N(sign * m, eta * s2),
    # sign = +-1 with prob 1/2 per edge. 0.0 is the original prior; the exact
    # oracle must use the same value.
    weight_mean: float = 0.0
    # IG shape of root variances for linear_iw (default 2.5) and linear_nig
    # (default 2.5); non-roots get alpha0 + |pa| / 2. Unused by the legacy keys.
    alpha0: float | None = None
    # Class probabilities of graph_type [MOTIF3] (D = 3 only): keys empty, single,
    # chain, fork, collider, complete (see datasets/motifs.py); the graph is
    # uniform within its class. Required for MOTIF3, invalid otherwise.
    motif_probs: dict[str, float] | None = None
    # P(K = 0), i.e. observation-only contexts. None keeps the truncated
    # geometric over {0..k_cap}; otherwise K >= 1 follows its shape.
    p_k0: float | None = None

    def __post_init__(self):
        """Validate the config and normalise types after JSON/YAML loading."""
        if self.function_generator not in REGISTRY:
            raise ValueError(
                f"{self.function_generator} is not a valid function generator. "
                f"Valid options are: {', '.join(REGISTRY)}."
            )
        self.num_variables = [int(d) for d in self.num_variables]
        self.graph_type = list(self.graph_type)
        # JSON/YAML round-trips turn int dict keys into strings.
        if isinstance(self.graph_degrees, dict):
            self.graph_degrees = {
                int(k): [int(v) for v in vs]
                for k, vs in self.graph_degrees.items()
            }
        else:
            self.graph_degrees = [int(v) for v in self.graph_degrees]
        # Config files give lists; keep tuples so configs compare/hash cleanly.
        self.n_ctx_range = tuple(self.n_ctx_range)  # type: ignore[assignment]
        self.dirichlet_alpha_range = tuple(self.dirichlet_alpha_range)  # type: ignore[assignment]
        self.do_scale_range = tuple(self.do_scale_range)  # type: ignore[assignment]
        self.wide_do_range = tuple(self.wide_do_range)  # type: ignore[assignment]
        try:
            self.qry_type_probs = tuple(float(p) for p in self.qry_type_probs)  # type: ignore[assignment]
        except ValueError as e:
            # YAML reads `1/3` as the string "1/3"; write 0.3333 instead.
            raise ValueError(
                f"qry_type_probs entries must be numbers, got "
                f"{self.qry_type_probs!r} (YAML reads 1/3 as a string)"
            ) from e
        if min(self.num_variables) < 1:
            raise ValueError("num_variables must be >= 1")
        # With no interventional regimes allowed, the context can only be
        # non-empty if the observational regime can be drawn.
        if self.k_max < 1 and self.p_obs <= 0:
            raise ValueError("p_obs > 0 is required when k_max < 1")
        if self.n_ref < 2 and self.normalise:
            raise ValueError("n_ref >= 2 is required when normalise=True")
        if len(self.qry_type_probs) != 3 or sum(self.qry_type_probs) <= 0:
            raise ValueError("qry_type_probs must be 3 non-negative weights")
        if not self.max_abs_value > 0:
            raise ValueError("max_abs_value must be > 0")
        if self.n_obs_min < 0:
            raise ValueError("n_obs_min must be >= 0")
        self.weight_mean = float(self.weight_mean)
        if not self.weight_mean >= 0:
            raise ValueError("weight_mean must be >= 0")
        if (
            self.weight_mean != 0.0
            and not is_linear_gaussian(self.function_generator)
        ):
            raise ValueError(
                "weight_mean is only supported for linear-Gaussian families"
            )
        self.weight_scale = float(self.weight_scale)
        if not self.weight_scale > 0:
            raise ValueError("weight_scale must be > 0")
        if (
            self.weight_scale != 1.0
            and not is_linear_gaussian(self.function_generator)
        ):
            raise ValueError(
                "weight_scale is only supported for linear-Gaussian families"
            )
        if self.alpha0 is not None:
            self.alpha0 = float(self.alpha0)
            if not self.alpha0 > 0:
                raise ValueError("alpha0 must be > 0")
            if is_legacy_key(self.function_generator) or not is_linear_gaussian(
                self.function_generator
            ):
                raise ValueError("alpha0 is only supported for linear_iw and linear_nig")
        if is_linear_gaussian(self.function_generator):
            # Raises for an invalid combination (e.g. linear_iw with weight_scale != 1).
            LinearGaussianPrior.from_config(self)
        if self.p_k0 is not None:
            self.p_k0 = float(self.p_k0)
            if not 0.0 <= self.p_k0 < 1.0:
                raise ValueError("p_k0 must be in [0, 1)")
        is_motif = "MOTIF3" in self.graph_type
        if is_motif and (
            self.graph_type != ["MOTIF3"] or self.num_variables != [3]
        ):
            raise ValueError("MOTIF3 needs graph_type: [MOTIF3] and num_variables: [3]")
        if is_motif != (self.motif_probs is not None):
            raise ValueError("motif_probs is required for, and only for, MOTIF3")
        if self.motif_probs is not None:
            self.motif_probs = validate_motif_probs(self.motif_probs)
        if "ER_CC" in self.graph_type:
            degs = (
                [v for vs in self.graph_degrees.values() for v in vs]
                if isinstance(self.graph_degrees, dict)
                else self.graph_degrees
            )
            if min(degs) < 2 or min(self.num_variables) < 3:
                raise ValueError("ER_CC needs >= 2 edges and D >= 3 (a common cause)")
        # A task has at most R = min(k_max, D) + 1 regimes and reserves n_obs_min
        # observational rows; requiring n_ctx_range[0] >= max R + n_obs_min keeps
        # the n_ctx floor a no-op, so a fixed per-batch n_ctx gives equal sizes.
        max_regimes = min(self.k_max, max(self.num_variables)) + 1
        if self.n_ctx_range[0] < max_regimes + self.n_obs_min:
            raise ValueError(
                f"n_ctx_range[0]={self.n_ctx_range[0]} must be >= "
                f"min(k_max, max(num_variables)) + 1 + n_obs_min = "
                f"{max_regimes + self.n_obs_min}"
            )


        if self.qry_mode not in ("per_row", "grouped"):
            raise ValueError(
                f"qry_mode must be 'per_row' or 'grouped', got {self.qry_mode!r}"
            )
        self.qry_groups = [int(g) for g in self.qry_groups]
        if self.qry_mode == "grouped":
            self._validate_grouped()

    def _validate_grouped(self) -> None:
        """Checks that only apply in grouped query mode."""
        if min(self.num_variables) < 2:
            raise ValueError("grouped qry_mode needs min(num_variables) >= 2")
        if not self.qry_groups or min(self.qry_groups) < 1:
            raise ValueError("qry_groups must be a non-empty list of ints >= 1")
        if max(self.qry_groups) > min(self.num_variables):
            raise ValueError(
                f"max(qry_groups)={max(self.qry_groups)} must be <= "
                f"min(num_variables)={min(self.num_variables)}: the treatments "
                "of a task's groups are distinct"
            )
        bad = [g for g in self.qry_groups if self.n_qry % g != 0]
        if bad:
            raise ValueError(
                f"n_qry={self.n_qry} must be divisible by every G in qry_groups "
                f"(not by {bad})"
            )
        if sum(self.qry_type_probs[:2]) <= 0:
            raise ValueError(
                "grouped qry_mode needs qry_type_probs[0] + qry_type_probs[1] > 0"
            )


@dataclass
class Design:
    """What is measured in one task, in UNPERMUTED variable indices.

    Args:
    ----------
    num_variables : int
        D.

    regime_targets : list[tuple[int, ...]]
        One INTERVENTION target set per context regime; () is the
        observational regime.

    regime_sizes : list[int]
        Number of context rows n_r per regime (same length, all >= 1).

    do_dist : dict
        {"kind": "normal", "scale": s} or {"kind": "uniform", "width": w}.
        Every intervened cell gets its own independent draw.

    qry_targets : list[tuple[int, ...]]
        One INTERVENTION target set per query row; () is an observational
        query. The cells to predict are the remaining (non-intervened) ones.

    alpha : float | None
        Dirichlet concentration used for regime_sizes (bookkeeping only).

    n_groups, group_treatments, group_types :
        Grouped query mode only: G, the treatment of each group (unpermuted)
        and whether it is "seen" or "unseen" in the context. 0 / [] / [] in
        per-row mode.

    regime_do_values : list[float | None] | None
        Optional, one entry per context regime: every intervened cell of that
        regime takes this value instead of a draw from do_dist. None (default)
        keeps the usual random do-values.
    """

    num_variables: int
    regime_targets: list[tuple[int, ...]]
    regime_sizes: list[int]
    do_dist: dict
    qry_targets: list[tuple[int, ...]]
    # Dirichlet concentration used for regime_sizes (bookkeeping only).
    alpha: float | None = None
    n_groups: int = 0
    group_treatments: list[int] = field(default_factory=list)
    group_types: list[str] = field(default_factory=list)
    # Optional fixed do-value per context regime (None = draw from do_dist as
    # usual). Used by diagnostics such as plot_obs_vs_int; default off.
    regime_do_values: list[float | None] | None = None

    def __post_init__(self):
        """Cast to plain python ints/tuples and check sizes and target ranges."""
        self.regime_targets = [
            tuple(int(j) for j in t) for t in self.regime_targets
        ]
        self.qry_targets = [tuple(int(j) for j in t) for t in self.qry_targets]
        self.regime_sizes = [int(n) for n in self.regime_sizes]
        if len(self.regime_targets) != len(self.regime_sizes):
            raise ValueError("regime_targets and regime_sizes differ in length")
        if len(self.regime_sizes) == 0 or min(self.regime_sizes) < 1:
            raise ValueError("Context needs >= 1 regime, each with n_r >= 1")
        D = self.num_variables
        for t in self.regime_targets + self.qry_targets:
            if any(j < 0 or j >= D for j in t):
                raise ValueError(f"Target {t} out of range for D={D}")
        if self.regime_do_values is not None and len(self.regime_do_values) != len(
            self.regime_targets
        ):
            raise ValueError("regime_do_values and regime_targets differ in length")


@dataclass
class Task:
    """One generated task: unpadded numpy arrays, variables already permuted.

    Args:
    ----------
    ctx_x : np.ndarray [Nc, D] float32
        Context values; intervened cells hold their do-value.

    ctx_int : np.ndarray [Nc, D] bool
        True where the cell was intervened on.

    ctx_regime : np.ndarray [Nc] int64
        Index into `regime_targets` for each context row. Not passed to the
        model in the flat design; the model sees the same information through
        ctx_int.

    qry_x : np.ndarray [Nq, D] float32
        TRUE query values, including the cells to predict. Masking the targets
        is the model's job.

    qry_int : np.ndarray [Nq, D] bool
        Do-cells of each query row.

    qry_tgt : np.ndarray [Nq, D] bool
        Cells to predict (= ~qry_int).

    e.g. qry_x = [3,4,9], qry_int = [0,1,0], qry_tgt = [1,0,1]
    means the entry 0 is a target and its true value is 3,
    entry 1 is intervened on and its value is 4,
    and entry 2 is a target and its true value is 9.

    graph : np.ndarray [D, D] int8
        graph[i, j] = 1 iff i -> j. Diagnostics only, never a model input.

    regime_targets : list[tuple[int, ...]]
        INTERVENTION target set of each context regime, in permuted variable
        indices. () is the observational regime. ctx_regime[i] indexes into this
        list, so context row i was generated under do(X_t) for t in
        regime_targets[ctx_regime[i]], with a row-specific do-value per target.
        Example: regime_targets = [(), (2,)] and ctx_regime = [1, 0] means row 0
        has X_2 set by intervention and row 1 is observational.
        Prediction targets are unrelated to regimes: they are qry_tgt = ~qry_int.
        Not passed to the model in the flat design; the model sees the same
        information through ctx_int.

    meta : dict
        Family, graph type and s0, design parameters, regime sizes, query types
        and targets, seed_key, used_key, resample count, max_abs, device, and
        n_groups / group_treatments (permuted) / group_types (grouped mode;
        0 / [] / [] in per-row mode).

    qry_group : np.ndarray [Nq] int64
        Question group of each query row in grouped mode:
        repeat(arange(G), Nq // G), so groups are contiguous blocks. All -1 in
        per-row mode (and for tasks loaded from older chunks).
    """

    ctx_x: np.ndarray
    ctx_int: np.ndarray
    ctx_regime: np.ndarray
    qry_x: np.ndarray
    qry_int: np.ndarray
    qry_tgt: np.ndarray
    graph: np.ndarray
    regime_targets: list[tuple[int, ...]]
    meta: dict = field(default_factory=dict)
    qry_group: np.ndarray | None = None

    def __post_init__(self):
        # Default: per-row mode (and tasks from chunks written before groups).
        if self.qry_group is None:
            self.qry_group = np.full(self.qry_x.shape[0], -1, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #


def _sample_expected_edges(cfg: MixedRegimeConfig, D: int) -> int:
    """`sample_uniform_expected_degree` semantics: list or dict keyed by D."""
    if isinstance(cfg.graph_degrees, dict):
        if D not in cfg.graph_degrees:
            raise KeyError(f"graph_degrees has no entry for D={D}")
        possible = cfg.graph_degrees[D]
    else:
        possible = cfg.graph_degrees
    return int(np.random.choice(possible))


_MOTIF_DAGS: dict | None = None


def _motif_dags() -> dict:
    """Cached class -> labelled DAGs lists of `motifs.motif_dags`."""
    global _MOTIF_DAGS
    if _MOTIF_DAGS is None:
        _MOTIF_DAGS = motif_dags()
    return _MOTIF_DAGS


def sample_graph(cfg: MixedRegimeConfig, D: int) -> tuple[np.ndarray, dict]:
    """Sample a DAG with graph[i, j] = 1 iff i -> j.

    Uses the global numpy RNG and Python's `random` (via igraph), so call it
    after seeding. Index order is NOT guaranteed to be topological (e.g. "FC"),
    so always use `topological_order` on the result. "ER_CC" is "ER" redrawn
    until some node has >= 2 children (a common cause). With `motif_probs`
    ("MOTIF3", D = 3) the class is drawn from `motif_probs` and the graph is
    uniform within it.

    Returns:
    ----------
    (graph, info) where info = {"graph_type": str, "s0": int} records the graph
    type and the s0 actually passed to `generate_synthetic_dag` (after capping).
    """
    if cfg.motif_probs is not None:
        # MOTIF3: class from motif_probs, then a uniform labelled DAG in the class.
        probs = [cfg.motif_probs[c] for c in MOTIF_CLASSES]
        cls = MOTIF_CLASSES[int(np.random.choice(len(MOTIF_CLASSES), p=probs))]
        dags = _motif_dags()[cls]
        graph = dags[int(np.random.randint(len(dags)))]
        return graph.copy(), {"graph_type": "MOTIF3", "s0": int(graph.sum())}
    s0 = _sample_expected_edges(cfg, D)
    graph_type = str(np.random.choice(cfg.graph_type))
    if graph_type in ("ER", "ER_CC"):
        # igraph errors if m exceeds the number of possible edges.
        s0 = min(s0, D * (D - 1) // 2)
    elif graph_type == "SF" and round(s0 / D) < 1:
        # SF adds m = round(s0 / D) edges per new node, so a small s0 would
        # silently give an empty graph; use m = 1 instead.
        s0 = D
    if graph_type == "ER_CC":
        # ER conditioned on a common cause: some node has >= 2 children.
        for _ in range(1000):
            graph = generate_synthetic_dag(d=D, s0=s0, graph_type="ER")
            if (np.asarray(graph) != 0).sum(axis=1).max() >= 2:
                break
        else:
            raise RuntimeError(f"ER_CC: no common cause in 1000 draws (D={D}, s0={s0})")
    else:
        graph = generate_synthetic_dag(d=D, s0=s0, graph_type=graph_type)
    info = {"graph_type": graph_type, "s0": int(s0)}
    return (np.asarray(graph) != 0).astype(np.int8), info


def topological_order(graph: np.ndarray) -> list[int]:
    """Kahn's algorithm (smallest index first). Raises if graph is cyclic."""
    adj = np.asarray(graph) != 0
    D = adj.shape[0]
    # In-degree = number of parents (column sums, since adj[i, j] means i -> j).
    indeg = adj.sum(axis=0).astype(int)
    ready = sorted(np.where(indeg == 0)[0].tolist())
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in np.where(adj[i])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(int(j))
        # Keep the order deterministic for a given graph.
        ready.sort()
    if len(order) != D:
        raise ValueError("Graph is not a DAG")
    return order


def descendants(graph: np.ndarray, node: int) -> set[int]:
    """Strict descendants of `node` (excludes `node`); graph[i, j] = 1 iff i -> j."""
    adj = np.asarray(graph) != 0
    seen: set[int] = set()
    stack = [node]
    # Depth-first walk along outgoing edges.
    while stack:
        for j in np.where(adj[stack.pop()])[0]:
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return seen


# --------------------------------------------------------------------------- #
# Designs
# --------------------------------------------------------------------------- #


def _qry_type_weights(cfg: MixedRegimeConfig, n_seen: int, n_unseen: int):
    """Probabilities of (seen, unseen, empty) query types for this design.

    Types with no available node (no context targets -> no "seen"; every node
    targeted -> no "unseen") get probability 0 and the rest is renormalised.
    """
    w = np.array(cfg.qry_type_probs, dtype=np.float64)
    if n_seen == 0:
        w[0] = 0.0
    if n_unseen == 0:
        w[1] = 0.0
    if w.sum() <= 0:
        # Only the empty type is left (or its weight was 0 too): fall back to it.
        w = np.array([0.0, 0.0, 1.0])
    return w / w.sum()


def sample_design(
    cfg: MixedRegimeConfig,
    D: int,
    n_ctx: int | None = None,
    n_groups: int | None = None,
) -> Design:
    """Sample which regimes the context contains, their sizes, the do-value
    distribution and the query targets (spec section 6.1).

    Uses the global numpy RNG, so call it after seeding.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Prior hyperparameters.

    D : int
        Number of variables.

    n_ctx : int | None
        Total context rows. None draws it from cfg.n_ctx_range; given, it
        replaces that draw (one fewer RNG call), still floored at R.

    n_groups : int | None
        Grouped mode only: number of question groups G. None draws it from
        cfg.qry_groups (after all context-design draws).

    Returns:
    ----------
    Design
        In unpermuted variable indices.
    """
    # K ~ truncated geometric on {0, ..., min(k_max, D)}.
    k_cap = max(0, min(cfg.k_max, D))
    # larger p_geom makes K=0 more likely
    k_probs = (1.0 - cfg.p_geom) ** np.arange(k_cap + 1)
    if cfg.p_k0 is not None and k_cap > 0:
        # P(K = 0) = p_k0; K >= 1 keeps the geometric shape.
        k_probs[0] = 0.0
        k_probs = (1.0 - cfg.p_k0) * k_probs / k_probs.sum()
        k_probs[0] = cfg.p_k0
    k_probs = k_probs / k_probs.sum()
    # Resample until the context is non-empty (obs regime and/or K >= 1).
    while True:
        has_obs = cfg.n_obs_min > 0 or bool(np.random.rand() < cfg.p_obs)
        K = int(np.random.choice(k_cap + 1, p=k_probs))
        if has_obs or K > 0:
            break

    # K distinct single-node targets; the observational regime (if any) is first.
    targets = np.random.choice(D, size=K, replace=False).tolist() if K else []
    regime_targets: list[tuple[int, ...]] = [()] * has_obs + [
        (int(j),) for j in targets
    ]

    # Regime sizes: every regime gets 1 row, the rest is split by
    # Dirichlet-multinomial. Small alpha gives very unbalanced regimes.
    R = len(regime_targets)
    reserve = cfg.n_obs_min if has_obs else 0
    if n_ctx is None:
        lo, hi = cfg.n_ctx_range
        n_ctx = int(np.random.randint(lo, hi + 1))
    n_ctx = max(int(n_ctx), R + reserve)
    a_lo, a_hi = cfg.dirichlet_alpha_range
    alpha = float(np.exp(np.random.uniform(np.log(a_lo), np.log(a_hi))))
    p = np.random.dirichlet(alpha * np.ones(R))
    # Every regime gets one row and the observational regime (first) also gets
    # its reserved rows; the rest is split by a Dirichlet-multinomial.
    regime_sizes = (1 + np.random.multinomial(n_ctx - R - reserve, p)).tolist()
    if reserve:
        regime_sizes[0] += reserve

    # One do-value distribution per task, shared by context and query rows.
    if np.random.rand() < cfg.p_wide_do:
        do_dist = {
            "kind": "uniform",
            "width": float(np.random.uniform(*cfg.wide_do_range)),
        }
    else:
        do_dist = {
            "kind": "normal",
            "scale": float(np.random.uniform(*cfg.do_scale_range)),
        }

    # Query targets: "seen" = a node intervened on in the context, "unseen" =
    # any other node, "empty" = observational query.
    seen = sorted(int(j) for j in targets)
    unseen = sorted(set(range(D)) - set(seen))
    if cfg.qry_mode == "grouped":
        return _grouped_design(
            cfg, D, n_groups, seen, unseen, regime_targets, regime_sizes,
            do_dist, alpha,
        )  # fmt: skip
    if n_groups is not None:
        raise ValueError("n_groups is only valid in grouped qry_mode")
    w = _qry_type_weights(cfg, len(seen), len(unseen))
    types = np.random.choice(3, size=cfg.n_qry, p=w)
    qry_targets: list[tuple[int, ...]] = []
    for t in types:
        if t == 0:
            qry_targets.append((int(np.random.choice(seen)),))
        elif t == 1:
            qry_targets.append((int(np.random.choice(unseen)),))
        else:
            qry_targets.append(())

    return Design(
        num_variables=D,
        regime_targets=regime_targets,
        regime_sizes=regime_sizes,
        do_dist=do_dist,
        qry_targets=qry_targets,
        alpha=alpha,
    )


def _grouped_design(
    cfg: MixedRegimeConfig,
    D: int,
    n_groups: int | None,
    seen: list[int],
    unseen: list[int],
    regime_targets: list[tuple[int, ...]],
    regime_sizes: list[int],
    do_dist: dict,
    alpha: float,
) -> Design:
    """Grouped query mode: G groups with distinct single-node treatments.

    Called by sample_design after every context-design draw, so the context of
    a grouped task is drawn exactly like a per-row task with the same key.
    """
    G = int(np.random.choice(cfg.qry_groups)) if n_groups is None else n_groups
    if G < 1 or G > D or cfg.n_qry % G != 0:
        raise ValueError(
            f"n_groups={G} must be in [1, D={D}] and divide n_qry={cfg.n_qry}"
        )
    w = np.array(cfg.qry_type_probs[:2], dtype=np.float64)
    w = w / w.sum()
    pools = {"seen": seen, "unseen": unseen}
    used: set[int] = set()
    treatments: list[int] = []
    types: list[str] = []
    for _ in range(G):
        kind = ("seen", "unseen")[int(np.random.choice(2, p=w))]
        pool = [j for j in pools[kind] if j not in used]
        if not pool:
            # This type is exhausted; use the other one.
            kind = "unseen" if kind == "seen" else "seen"
            pool = [j for j in pools[kind] if j not in used]
        if not pool:
            raise ValueError(f"No treatment left for {G} groups with D={D}")
        t = int(np.random.choice(pool))
        used.add(t)
        treatments.append(t)
        types.append(kind)
    rows_per_group = cfg.n_qry // G
    # Contiguous blocks of query rows, in group order.
    qry_targets: list[tuple[int, ...]] = [
        (t,) for t in treatments for _ in range(rows_per_group)
    ]
    return Design(
        num_variables=D,
        regime_targets=regime_targets,
        regime_sizes=regime_sizes,
        do_dist=do_dist,
        qry_targets=qry_targets,
        alpha=alpha,
        n_groups=G,
        group_treatments=treatments,
        group_types=types,
    )


# for evaluation
def classify_qry_targets(
    regime_targets: list[tuple[int, ...]], qry_targets: list[tuple[int, ...]]
) -> list[str]:
    """Label each query target set as 'seen', 'unseen' or 'empty'.

    Derived from the targets rather than stored by `sample_design`, so it also
    works for hand-written designs.
    """
    ctx_nodes = {j for t in regime_targets for j in t}
    out = []
    for t in qry_targets:
        if len(t) == 0:
            out.append("empty")
        elif set(t) <= ctx_nodes:
            out.append("seen")
        else:
            out.append("unseen")
    return out


def _sample_do_values(do_dist: dict, size: int) -> np.ndarray:
    """Draw `size` iid do-values (in standardised space when normalise=True)."""
    if do_dist["kind"] == "normal":
        return np.random.normal(0.0, do_dist["scale"], size=size)
    if do_dist["kind"] == "uniform":
        w = do_dist["width"]
        return np.random.uniform(-w, w, size=size)
    raise ValueError(f"Unknown do_dist kind {do_dist['kind']}")


def _standardise(
    x: np.ndarray, mean: float, std: float, eps: float = 1e-6
) -> np.ndarray:
    """(x - mean) / std, with std floored at eps (constant variables)."""
    return (x - mean) / max(float(std), eps)


# --------------------------------------------------------------------------- #
# Task generation
# --------------------------------------------------------------------------- #


def _generate_once(
    cfg: MixedRegimeConfig,
    key: tuple[int, ...],
    graph: np.ndarray | None,
    design: Design | None,
    num_variables: int | None,
    n_ctx_override: int | None = None,
    n_groups: int | None = None,
):
    """One generation attempt in unpermuted variable order.

    `key` is the 7-tuple make_key(...) + (attempt,). Returns the full row
    buffer (reference rows included) plus everything `generate_task` needs to
    permute, split and package it. The caller checks the buffer and retries
    with the next attempt if needed.
    """
    _seed_all(key)

    # Determine the number of variables D. We accept several ways to
    # specify D so this function is flexible for different calling contexts:
    # These sources may differ because sometimes the user wants to fix the
    # causal graph or design up-front, other times they only want to fix the
    # variable count, and in other cases generation should draw from the
    # configured prior. The priority order ensures consistency when multiple
    # specifications are available (e.g. a design must be respected over a
    # loose graph or prior).

    # - If a Design is provided, it explicitly encodes the number of
    #   variables (design.num_variables) and takes highest priority because
    #   the design also contains regime sizes, targets and do-values that
    #   must match D.
    if design is not None:
        D = design.num_variables
    # - If no Design is given but a graph adjacency matrix is provided,
    #   infer D from the graph shape. This lets callers pass a concrete
    #   structure and have generation follow that variable set.

    elif graph is not None:
        D = int(np.asarray(graph).shape[0])
    # - If neither Design nor graph are present but num_variables is
    #   supplied, use that explicit integer.
    elif num_variables is not None:
        D = int(num_variables)
    # - Otherwise sample D from the prior specified in cfg.num_variables.
    #   This is the default case when no other information is available.
    else:
        D = int(np.random.choice(cfg.num_variables))

    meta: dict[str, Any] = {}
    if graph is None:
        graph, graph_info = sample_graph(cfg, D)
    else:
        # Explicit graph: nothing was sampled.
        graph_info = {"graph_type": None, "s0": None}
    meta.update(graph_info)
    graph = (np.asarray(graph) != 0).astype(np.int8)
    if graph.shape != (D, D):
        raise ValueError(f"graph shape {graph.shape} does not match D={D}")
    topo = topological_order(graph)

    if design is None:
        design = sample_design(cfg, D, n_ctx=n_ctx_override, n_groups=n_groups)

    # intervention_range_multiplier is unused here: do-values come from the
    # design's do_dist instead.
    prior = (
        LinearGaussianPrior.from_config(cfg)
        if is_linear_gaussian(cfg.function_generator)
        else None
    )
    if prior is not None and not is_legacy_key(cfg.function_generator):
        # New families receive the prior at construction.
        gen = REGISTRY[cfg.function_generator](
            num_variables=D, intervention_range_multiplier=1, prior=prior
        )
    else:
        gen = REGISTRY[cfg.function_generator](
            num_variables=D, intervention_range_multiplier=1
        )
    fdict = gen.generate_functions(graph)
    # Legacy keys only: patch eta / weight_mean onto the mechanisms (kept so the
    # random stream and the goldens are unchanged).
    if cfg.weight_scale != 1.0 and is_legacy_key(cfg.function_generator):
        # LinearNonIdentifiable draws w ~ N(0, eta * var1) at its call; eta is
        # read only there, so no random numbers change before that point.
        for f in fdict.values():
            f.eta = cfg.weight_scale
    if cfg.weight_mean != 0.0 and is_legacy_key(cfg.function_generator):
        # Signed-mean mixture prior: LinearNonIdentifiable adds sign * weight_mean
        # (sign = +-1 per edge) to w at its call. The sign is drawn only when
        # weight_mean != 0, so default configs keep the same random numbers.
        for f in fdict.values():
            f.weight_mean = cfg.weight_mean

    # Row layout: [ref | ctx regime 0 | ... | ctx regime R-1 | qry]
    n_ref = cfg.n_ref
    n_ctx = int(sum(design.regime_sizes))
    n_qry = len(design.qry_targets)
    n_tot = n_ref + n_ctx + n_qry
    if cfg.function_generator in GP_FAMILIES and n_tot > cfg.max_total_rows_gp:
        raise ValueError(
            f"N_tot={n_tot} exceeds max_total_rows_gp={cfg.max_total_rows_gp} "
            f"for GP family {cfg.function_generator}"
        )

    # Build the intervention mask row by row: regime r's targets are set on
    # its n_r context rows, each query row gets its own targets. Reference
    # rows are never intervened on.
    X = np.zeros((n_tot, D), dtype=np.float64)
    int_mask = np.zeros((n_tot, D), dtype=bool)
    ctx_regime = np.repeat(
        np.arange(len(design.regime_sizes), dtype=np.int64), design.regime_sizes
    )
    row = n_ref
    for targets, n_r in zip(design.regime_targets, design.regime_sizes):
        for j in targets:
            int_mask[row : row + n_r, j] = True
        row += n_r
    for q, targets in enumerate(design.qry_targets):
        for j in targets:
            int_mask[n_ref + n_ctx + q, j] = True
    # Independent do-value per intervened cell.
    int_val = np.zeros((n_tot, D), dtype=np.float64)
    int_val[int_mask] = _sample_do_values(design.do_dist, int(int_mask.sum()))
    # Fixed do-values overwrite the draws (the draws still happen, so the RNG
    # stream is the same as without them).
    if design.regime_do_values is not None:
        row = n_ref
        for targets, n_r, v in zip(
            design.regime_targets, design.regime_sizes, design.regime_do_values
        ):
            if v is not None:
                for j in targets:
                    int_val[row : row + n_r, j] = v
            row += n_r

    # Per-node mechanism and standardisation constants, for diagnostics only
    # (returned through generate_task(..., return_debug=True)).
    mechanisms: dict[int, dict] = {}
    for i in topo:
        pa = np.where(graph[:, i])[0]
        # ONE call on ALL rows (see module docstring). Rows where i is
        # intervened also go through the mechanism; their outputs are
        # discarded below (for GPs this marginalises those points out).

        # evalutate the mechanism function for variable i
        # using the parent values for all rows (reference, context, query)
        out = np.asarray(fdict[i](X[:, pa]), dtype=np.float64).reshape(n_tot)
        loc, scale = 0.0, 1.0
        if cfg.normalise:
            # Statistics from the hidden observational reference rows only, so
            # the SCM does not depend on the design. Children then see
            # standardised parents, as in the original generator.
            ref = out[:n_ref]
            loc, scale = float(ref.mean()), max(float(ref.std()), 1e-6)
            out = _standardise(out, ref.mean(), ref.std())
        mechanisms[int(i)] = {
            "parents": pa.tolist(),
            "mechanism": fdict[i],
            "loc": loc,
            "scale": scale,
        }
        # Intervened cells take their do-value, so descendants see it.
        X[:, i] = np.where(int_mask[:, i], int_val[:, i], out)

    meta["_mechanisms"] = mechanisms  # popped in generate_task
    meta.update(
        {
            "family": cfg.function_generator,
            "D": D,
            "K": sum(1 for t in design.regime_targets if len(t) > 0),
            "has_obs": any(len(t) == 0 for t in design.regime_targets),
            "regime_sizes": list(design.regime_sizes),
            "do_dist": dict(design.do_dist),
            "alpha": design.alpha,
            "qry_types": classify_qry_targets(
                design.regime_targets, design.qry_targets
            ),
            "normalise": cfg.normalise,
        }
    )
    if prior is not None:
        meta["prior"] = dataclasses.asdict(prior)
        meta["is_iw_compatible"] = prior.is_iw_compatible
    return (
        X,
        int_mask,
        int_val,
        ctx_regime,
        graph,
        design,
        topo,
        n_ref,
        n_ctx,
        meta,
    )


def generate_task(
    cfg: MixedRegimeConfig,
    seed: int | tuple[int, ...],
    graph: np.ndarray | None = None,
    design: Design | None = None,
    num_variables: int | None = None,
    return_debug: bool = False,
    n_ctx: int | None = None,
    n_groups: int | None = None,
    permute: bool = True,
):
    """
    Generate one task. `graph` / `design` are sampled if not given;
    `num_variables` fixes D when neither is given.

    The same key (and the same explicit graph/design) always gives the same
    task on the same device type (CPU vs CUDA streams differ for GP/NN
    families). If the data are non-finite or exceed cfg.max_abs_value, the
    whole task is regenerated from key + (attempt,); an explicit graph/design
    is kept across attempts.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Prior hyperparameters.

    seed : int | tuple[int, ...]
        A 6-tuple from `make_key`, or an int i, which is shorthand for
        make_key(SOURCE_ADHOC, 0, 0, i) (tests and ad hoc calls).

    graph : np.ndarray | None
        [D, D] DAG with graph[i, j] = 1 iff i -> j, in unpermuted indices.

    design : Design | None
        Design in unpermuted indices.

    num_variables : int | None
        Fixes D (e.g. one D per batch) when graph and design are not given.

    n_ctx : int | None
        Fixes the total number of context rows (e.g. one N_ctx per batch).
        Only valid when `design` is not given. None keeps the prior's draw and
        the exact RNG consumption of earlier versions.

    n_groups : int | None
        Grouped qry_mode only: fixes the number of question groups G (e.g. one
        G per batch). Only valid when `design` is not given.

    return_debug : bool
        Also return a debug dict.
        debug["mechanisms"] maps each unpermuted node index to {"parents",
        "mechanism", "loc", "scale"}: the mechanism object after its single
        call and the reference standardisation (0, 1 when normalise=False).
        Combine with debug["perm"] to map to the task's permuted indices.

    permute : bool
        Randomly permute the variables (default). False keeps the input
        labels, so an explicit graph/design is returned in its own indices.

    Returns:
    ----------
    Task, or (Task, debug_dict) if return_debug=True. debug_dict holds the
    do-value buffers (after permutation / shuffling, aligned with the returned
    arrays), the variable permutation, the topological order (unpermuted) and
    the row counts.
    """
    if isinstance(seed, (int, np.integer)):
        key = make_key(SOURCE_ADHOC, 0, 0, int(seed))
    else:
        if len(seed) != 6:
            raise ValueError(
                f"seed key must be a 6-tuple from make_key, got {seed}"
            )
        key = make_key(*seed)

    if design is not None and n_ctx is not None:
        raise ValueError("Pass either an explicit design or n_ctx, not both")
    if n_groups is not None:
        if cfg.qry_mode != "grouped":
            raise ValueError("n_groups is only valid in grouped qry_mode")
        if design is not None:
            raise ValueError(
                "Pass either an explicit design or n_groups, not both"
            )

    for attempt in range(cfg.max_resample_attempts + 1):
        res = _generate_once(
            cfg, key + (attempt,), graph, design, num_variables, n_ctx, n_groups
        )
        X = res[0]
        # The bound keeps values far below the float32 max, so the cast at the
        # end cannot overflow to inf.
        finite = bool(np.isfinite(X).all())
        ok = finite and np.abs(X).max() <= cfg.max_abs_value
        if ok:
            break
        RESAMPLE_COUNTS["nonfinite" if not finite else "too_large"] += 1
    else:
        raise RuntimeError(
            f"Task with key {key} still non-finite or exceeding max_abs_value="
            f"{cfg.max_abs_value} after {cfg.max_resample_attempts} resamples"
        )
    X, int_mask, int_val, ctx_regime, g, des, topo, n_ref, n_ctx, meta = res
    mechanisms = meta.pop("_mechanisms")  # never stored in Task.meta
    # Computed on the rows that are returned (reference rows are dropped).
    max_abs = float(np.abs(X[n_ref:]).max())
    D = des.num_variables

    # Variable permutation, as in the original generator: new column k holds
    # old variable perm[k]; old index j moves to inv[j].
    perm = np.random.permutation(D)  # drawn even when unused: same RNG stream
    if not permute:
        perm = np.arange(D)
    inv = np.argsort(perm)
    X, int_mask, int_val = X[:, perm], int_mask[:, perm], int_val[:, perm]
    g = g[perm][:, perm]
    regime_targets = [tuple(int(inv[j]) for j in t) for t in des.regime_targets]
    qry_targets = [tuple(int(inv[j]) for j in t) for t in des.qry_targets]

    # Drop reference rows, split, shuffle context rows (one permutation for
    # all context arrays, so regimes are not contiguous). Query order is kept.
    ctx = slice(n_ref, n_ref + n_ctx)
    qry = slice(n_ref + n_ctx, None)
    shuffle = np.random.permutation(n_ctx)
    ctx_x = X[ctx][shuffle]
    ctx_int = int_mask[ctx][shuffle]
    ctx_val = int_val[ctx][shuffle]
    ctx_regime = ctx_regime[shuffle]
    qry_int = int_mask[qry]

    meta.update(
        {
            "seed_key": list(key),
            "used_key": list(key) + [attempt],
            "resample_attempts": attempt,
            "qry_targets": qry_targets,
            "n_groups": des.n_groups,
            "group_treatments": [int(inv[t]) for t in des.group_treatments],
            "group_types": list(des.group_types),
            "max_abs": max_abs,
            # GP/NN mechanisms draw from the CUDA stream when it is available.
            "device": (
                "cpu"
                if cfg.function_generator in CPU_ONLY_FAMILIES
                or not torch.cuda.is_available()
                else "cuda"
            ),
        }
    )
    task = Task(
        ctx_x=ctx_x.astype(np.float32),
        ctx_int=ctx_int.astype(bool),
        ctx_regime=ctx_regime.astype(np.int64),
        qry_x=X[qry].astype(np.float32),
        qry_int=qry_int.astype(bool),
        qry_tgt=~qry_int,
        graph=g.astype(np.int8),
        regime_targets=regime_targets,
        meta=meta,
        # Query rows are never shuffled, so group blocks stay contiguous.
        qry_group=(
            np.repeat(
                np.arange(des.n_groups, dtype=np.int64),
                qry_int.shape[0] // des.n_groups,
            )
            if des.n_groups > 0
            else None
        ),
    )
    if not return_debug:
        return task
    debug = {
        "ctx_int_val": ctx_val,
        "qry_int_val": int_val[qry],
        "perm": perm,
        "topo": topo,
        "n_ref": n_ref,
        "n_tot": X.shape[0],
        "mechanisms": mechanisms,  # unpermuted node index -> mechanism record
    }
    return task, debug


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #


def task_from_dict(d: dict) -> Task:
    """Rebuild a Task from `dataclasses.asdict(task)` (the chunk format).

    Chunks written before grouped mode have no "qry_group"; Task then
    defaults it to all -1 (per-row).
    """
    return Task(**d)


def collate_tasks(tasks: list[Task], pad_D: int | None = None) -> dict[str, Any]:
    """Pad a list of tasks into a batch. Padding masks: True = padded.

    Rows are padded to the batch maxima Nc, Nq; variables to the batch max D or
    `pad_D`. Padded values are 0, padded flags are False, and `ctx_regime` is
    -1 on padded rows. `qry_tgt` is False on padded rows and variables.

    Args:
    ----------
    tasks : list[Task]
        Tasks to batch.

    pad_D : int | None
        Pad variables to this size instead of the batch max.

    Returns:
    ----------
    dict
        Tensors ctx_x, ctx_int, ctx_pad, qry_x, qry_int, qry_tgt, qry_pad,
        var_pad, graph, ctx_regime, qry_group ([B, Nq] int64, -1 in per-row
        mode and in padding) (shapes in the README) and "meta", the list of
        per-task meta dicts.
    """
    B = len(tasks)
    Ds = [t.ctx_x.shape[1] for t in tasks]
    D = max(Ds) if pad_D is None else int(pad_D)
    if D < max(Ds):
        raise ValueError(f"pad_D={pad_D} smaller than max task D={max(Ds)}")
    Nc = max(t.ctx_x.shape[0] for t in tasks)
    Nq = max(t.qry_x.shape[0] for t in tasks)

    # Start fully padded and fill in each task's valid block.
    ctx_x = np.zeros((B, Nc, D), dtype=np.float32)
    ctx_int = np.zeros((B, Nc, D), dtype=bool)
    ctx_pad = np.ones((B, Nc), dtype=bool)
    qry_x = np.zeros((B, Nq, D), dtype=np.float32)
    qry_int = np.zeros((B, Nq, D), dtype=bool)
    qry_tgt = np.zeros((B, Nq, D), dtype=bool)
    qry_pad = np.ones((B, Nq), dtype=bool)
    var_pad = np.ones((B, D), dtype=bool)
    graph = np.zeros((B, D, D), dtype=np.int8)
    ctx_regime = np.full((B, Nc), -1, dtype=np.int64)
    qry_group = np.full((B, Nq), -1, dtype=np.int64)

    for b, t in enumerate(tasks):
        nc, d = t.ctx_x.shape
        nq = t.qry_x.shape[0]
        ctx_x[b, :nc, :d] = t.ctx_x
        ctx_int[b, :nc, :d] = t.ctx_int
        ctx_pad[b, :nc] = False
        qry_x[b, :nq, :d] = t.qry_x
        qry_int[b, :nq, :d] = t.qry_int
        qry_tgt[b, :nq, :d] = t.qry_tgt
        qry_pad[b, :nq] = False
        var_pad[b, :d] = False
        graph[b, :d, :d] = t.graph
        ctx_regime[b, :nc] = t.ctx_regime
        qry_group[b, :nq] = t.qry_group

    out: dict[str, Any] = {
        k: torch.from_numpy(v)
        for k, v in {
            "ctx_x": ctx_x,
            "ctx_int": ctx_int,
            "ctx_pad": ctx_pad,
            "qry_x": qry_x,
            "qry_int": qry_int,
            "qry_tgt": qry_tgt,
            "qry_pad": qry_pad,
            "var_pad": var_pad,
            "graph": graph,
            "ctx_regime": ctx_regime,
            "qry_group": qry_group,
        }.items()
    }
    out["meta"] = [t.meta for t in tasks]
    return out


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def _worker_info() -> tuple[int, int]:
    """(worker_id, num_workers) inside a DataLoader worker, else (0, 1)."""
    info = torch.utils.data.get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


class MixedRegimeTaskDataset(IterableDataset):
    """
    On-the-fly generator yielding collated batches (use DataLoader with
    batch_size=None). Task key =
    make_key(SOURCE_STREAM, split_id, base_seed, epoch, worker_id, counter).

    By default every batch is padding-free: all its tasks share one D
    (same_variablenum_per_batch) and one context size N_ctx
    (same_nctx_per_batch), and N_qry is always cfg.n_qry. Both are drawn per
    batch from the SOURCE_STREAM_D key (D first, then N_ctx ~ U{n_ctx_range}),
    so each task's marginal prior is unchanged; only which tasks share a batch
    changes. In grouped qry_mode the number of question groups G is also drawn
    per batch from the same generator (after D and N_ctx), so every batch has
    one G.

    GP / NN mechanisms may move tensors to CUDA; with forked DataLoader workers
    use num_workers=0 for those families, or pre-generate chunks offline.

    Args:
    ----------
    cfg : MixedRegimeConfig
        Prior hyperparameters.

    batch_size : int
        Tasks per batch.

    base_seed : int
        Non-negative seed; part of every task key.

    same_variablenum_per_batch : bool
        Use one D per batch (no variable padding).

    iterations_per_epoch : int | None
        Batches per epoch, split across workers. None = infinite.

    split : str
        "train", "val" or "test" (see SPLIT_IDS). Different splits never share
        tasks, even with the same base_seed.

    fresh_each_epoch : bool
        False for validation and test, which must yield the same tasks on every
        pass.

    same_nctx_per_batch : bool
        Use one context size per batch (no context-row padding).

    Fresh tasks per epoch (fresh_each_epoch=True) work in all three DataLoader setups:
    - persistent workers (what CausalInferenceTrainer uses when num_workers > 0):
      each worker's copy keeps its counters, so later epochs continue with new keys;
    - num_workers=0: a single object, counters keep growing;
    - non-persistent workers: each epoch starts from a fresh copy (counters reset),
      so set_epoch(epoch) must be called each epoch (see SetEpochCallback).
    With fresh_each_epoch=False every pass yields the same tasks (validation/test).
    """

    def __init__(
        self,
        cfg: MixedRegimeConfig,
        batch_size: int,
        base_seed: int,
        same_variablenum_per_batch: bool = True,
        iterations_per_epoch: int | None = None,
        split: str = "train",
        fresh_each_epoch: bool = True,
        same_nctx_per_batch: bool = True,
    ):
        if base_seed < 0:
            raise ValueError("base_seed must be >= 0")
        self.cfg = cfg
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.same_variablenum_per_batch = same_variablenum_per_batch
        self.same_nctx_per_batch = same_nctx_per_batch
        self.iterations_per_epoch = iterations_per_epoch
        self.split_id = SPLIT_IDS[split]
        self.fresh_each_epoch = fresh_each_epoch
        self.epoch = 0
        # Key counters live on the object so that, with fresh_each_epoch, a
        # later pass over the same object (or persistent worker copy)
        # continues with new keys instead of replaying the first epoch.
        self._counter = 0
        self._batch_idx = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch; tasks differ between epochs. Call from the main process."""
        self.epoch = int(epoch)

    def __len__(self):
        if self.iterations_per_epoch is None:
            raise TypeError("Infinite dataset has no length")
        return self.iterations_per_epoch

    def __iter__(self):
        worker_id, num_workers = _worker_info()
        # Split the epoch's batches across workers (same as the original
        # InterventionDatasetGenerator); the first workers take the remainder.
        if self.iterations_per_epoch is None:
            iterations: int | None = None
        else:
            iterations = self.iterations_per_epoch // num_workers
            if worker_id < self.iterations_per_epoch % num_workers:
                iterations += 1

        # self.epoch is NOT advanced here: non-persistent DataLoader workers get
        # a fresh copy of the dataset each epoch, so an increment made in a
        # worker would be lost. The epoch comes from set_epoch.
        if not self.fresh_each_epoch:
            self._counter = self._batch_idx = 0
        done = 0  # batches yielded in this pass
        while iterations is None or done < iterations:
            D = None
            n_ctx = None
            n_groups = None
            grouped = self.cfg.qry_mode == "grouped"
            if (
                self.same_variablenum_per_batch
                or self.same_nctx_per_batch
                or grouped
            ):
                # Separate key source for the batch shape so it does not reuse
                # task streams. D is drawn first, then n_ctx.
                shape_key = make_key(
                    SOURCE_STREAM_D,
                    self.split_id,
                    self.base_seed,
                    self.epoch,
                    worker_id,
                    self._batch_idx,
                )
                rng = np.random.default_rng(list(shape_key))
                if self.same_variablenum_per_batch:
                    D = int(rng.choice(self.cfg.num_variables))
                if self.same_nctx_per_batch:
                    lo, hi = self.cfg.n_ctx_range
                    n_ctx = int(rng.integers(lo, hi + 1))
                if grouped:
                    # Drawn last, so per-row batches consume the same draws.
                    n_groups = int(rng.choice(self.cfg.qry_groups))
            tasks = []
            for _ in range(self.batch_size):
                key = make_key(
                    SOURCE_STREAM,
                    self.split_id,
                    self.base_seed,
                    self.epoch,
                    worker_id,
                    self._counter,
                )
                tasks.append(
                    generate_task(
                        self.cfg,
                        key,
                        num_variables=D,
                        n_ctx=n_ctx,
                        n_groups=n_groups,
                    )
                )
                self._counter += 1
            self._batch_idx += 1
            done += 1
            yield collate_tasks(tasks)


def list_chunks(chunk_dir: str) -> list[str]:
    """Sorted paths of the chunk_*.pt files in `chunk_dir`."""
    return sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pt")))


def load_chunk(path: str) -> list[Task]:
    """Load one chunk file as a list of Tasks."""
    # Chunks hold numpy arrays + python objects, so weights_only must be False.
    # Only load chunks you generated yourself.
    return [task_from_dict(d) for d in torch.load(path, weights_only=False)]


class MixedRegimeChunkDataset(IterableDataset):
    """
    Streams chunk files written by create_mixed_regime_chunks.py and yields
    collated batches (use DataLoader with batch_size=None). Chunk files are
    sharded across DataLoader workers. With shuffle=True, chunk order is
    shuffled per epoch and tasks pass through a shuffle buffer. With
    same_variablenum_per_batch=True, batches are bucketed by the exact shape
    (D, N_ctx), so they are padding-free when the chunks were written with
    --shape_group_size (N_qry is always cfg.n_qry).

    Requirements for full, padding-free batches and an exact len():
    - chunks written with shape groups, i.e. shape_group_size > 0 with
      chunk_size and num_tasks multiples of it (the writer enforces this), so
      every group lies in one chunk file and is read by one DataLoader worker;
    - shape_group_size % batch_size == 0, so every group splits into whole
      batches.
    __init__ warns when config.json shows these do not hold, and len() then
    raises TypeError (CausalInferenceTrainer needs an exact length to run
    validation on the last batch of an epoch).

    Use shuffle=False for validation and test. CausalInferenceTrainer runs
    validation many times per epoch, and reshuffling changes the batch
    composition, and with it the mean of per-batch losses.

    Args:
    ----------
    chunk_dir : str
        Directory containing chunk_*.pt files.

    batch_size : int
        Tasks per batch.

    shuffle : bool
        Shuffle chunk order and tasks. False streams tasks in file order.

    seed : int
        Seed for the shuffling.

    same_variablenum_per_batch : bool
        Only batch tasks with the same shape (D, N_ctx) together.

    shuffle_buffer_size : int
        Number of tasks held in the shuffle buffer.

    drop_last : bool
        Drop incomplete batches at the end of the epoch. With False, one
        partial batch per shape (D, N_ctx) is emitted at the end of each epoch,
        so batch sizes vary; use True for training if fixed batch sizes matter.
        With shape groups that are multiples of batch_size there are no
        partial batches.

    pad_D : int | None
        Passed to `collate_tasks`. Reintroduces variable padding, so it must
        stay None for the current encoder (it cannot mask padded variables).

    num_workers : int
        The DataLoader's num_workers. Only used by len() when
        same_variablenum_per_batch=False, where partial batches are counted per
        worker.

    The internal epoch counter advances in __iter__. That works with
    persistent workers (each worker's copy keeps counting) and with
    num_workers=0. With non-persistent workers each epoch gets a fresh copy,
    so call `set_epoch(epoch)` before each epoch (see SetEpochCallback).
    """

    def __init__(
        self,
        chunk_dir: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        same_variablenum_per_batch: bool = True,
        shuffle_buffer_size: int = 2048,
        drop_last: bool = False,
        pad_D: int | None = None,
        num_workers: int = 0,
    ):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.same_variablenum_per_batch = same_variablenum_per_batch
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last
        self.pad_D = pad_D
        self.num_workers = num_workers
        self.epoch = 0
        if not list_chunks(chunk_dir):
            raise FileNotFoundError(f"No chunk_*.pt files in {chunk_dir}")
        # Sizes from config.json (None if absent); used by __len__.
        self._num_tasks: int | None = None
        self._chunk_size: int | None = None
        self._group: int | None = None
        config_path = os.path.join(chunk_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                info = json.load(f)
            if info.get("device") == "cuda":
                warnings.warn(
                    f"Chunks in {chunk_dir} were generated with CUDA; they "
                    "are reproducible only on the same device type.",
                    stacklevel=2,
                )
            self._num_tasks = info.get("num_tasks")
            self._chunk_size = info.get("chunk_size")
            self._group = info.get("shape_group_size")
            group = self._group or 0
            if same_variablenum_per_batch and group == 0:
                warnings.warn(
                    f"{chunk_dir}: chunks were written without shape groups; "
                    "batches will need padding, which the current encoder "
                    "cannot mask",
                    stacklevel=2,
                )
            if group > 0 and group % batch_size != 0:
                warnings.warn(
                    f"{chunk_dir}: shape_group_size={group} is not a multiple "
                    f"of batch_size={batch_size}; each group ends in a partial "
                    "batch",
                    stacklevel=2,
                )

    def __len__(self) -> int:
        """Number of batches per epoch (summed over DataLoader workers).

        Exact when the chunks were written with shape groups (chunk_size and
        num_tasks multiples of shape_group_size, enforced by the writer) and
        shape_group_size % batch_size == 0: every group then lies in one chunk,
        is read by one worker, and splits into whole batches.

        With same_variablenum_per_batch=False (one bucket, padding allowed) the
        count is per worker (see num_workers), using this epoch's assignment of
        chunk files to workers. It is the same for every epoch when num_tasks
        is a multiple of chunk_size.
        """
        if self._num_tasks is None or self._group is None:
            raise TypeError(
                f"len() needs num_tasks and shape_group_size from "
                f"{self.chunk_dir}/config.json; regenerate the chunks with "
                "create_mixed_regime_chunks.py"
            )
        if not self.same_variablenum_per_batch:
            return sum(
                -(-n // self.batch_size)
                if not self.drop_last
                else n // self.batch_size
                for n in self._tasks_per_worker()
            )
        if self._group == 0 or self._group % self.batch_size != 0:
            raise TypeError(
                f"len() is not exact for {self.chunk_dir}: shape_group_size="
                f"{self._group} must be > 0 and a multiple of batch_size="
                f"{self.batch_size}; otherwise partial batches make the count "
                "depend on the shuffle"
            )
        return self._num_tasks // self.batch_size

    def _worker_paths(self, worker_id: int, num_workers: int) -> list[str]:
        """Chunk files read by one worker this epoch (same order as __iter__)."""
        paths = list_chunks(self.chunk_dir)
        # Chunk order must not depend on worker_id, so every worker shuffles
        # identically before taking its own slice.
        if self.shuffle:
            perm = np.random.default_rng([self.seed, self.epoch]).permutation(
                len(paths)
            )
            paths = [paths[i] for i in perm]
        return paths[worker_id::num_workers]

    def _tasks_per_worker(self) -> list[int]:
        """Task count per worker from config.json sizes, without loading files.

        All chunks hold chunk_size tasks except possibly the last one.
        """
        assert self._num_tasks is not None and self._chunk_size is not None
        all_paths = list_chunks(self.chunk_dir)
        last = all_paths[-1]
        last_size = self._num_tasks - (len(all_paths) - 1) * self._chunk_size
        nw = max(1, self.num_workers)
        return [
            sum(
                last_size if p == last else self._chunk_size
                for p in self._worker_paths(w, nw)
            )
            for w in range(nw)
        ]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the next iteration's shuffle."""
        self.epoch = epoch

    def _task_stream(self, rng: np.random.Generator, paths: list[str]):
        """Yield the tasks of `paths`, through a shuffle buffer if shuffling."""
        if not self.shuffle:
            for p in paths:
                yield from load_chunk(p)
            return
        # Fill the buffer, then emit a random element for each new task;
        # the leftovers are shuffled and emitted at the end.
        buffer: list[Task] = []
        for p in paths:
            for t in load_chunk(p):
                buffer.append(t)
                if len(buffer) >= self.shuffle_buffer_size:
                    k = int(rng.integers(len(buffer)))
                    buffer[k], buffer[-1] = buffer[-1], buffer[k]
                    yield buffer.pop()
        rng.shuffle(buffer)  # type: ignore[arg-type]
        yield from buffer

    def __iter__(self):
        worker_id, num_workers = _worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, worker_id])
        paths = self._worker_paths(worker_id, num_workers)
        self.epoch += 1

        # One bucket per exact shape (D, N_ctx), plus G in grouped mode (or a
        # single bucket); emit a batch when one is full.
        buckets: dict[tuple[int, ...], list[Task]] = defaultdict(list)
        for t in self._task_stream(rng, paths):
            key = _bucket_key(t) if self.same_variablenum_per_batch else (0, 0)
            buckets[key].append(t)
            if len(buckets[key]) == self.batch_size:
                yield collate_tasks(buckets.pop(key), pad_D=self.pad_D)
        if not self.drop_last:
            for key in sorted(buckets):
                yield collate_tasks(buckets[key], pad_D=self.pad_D)


def _bucket_key(t: Task) -> tuple[int, ...]:
    """Exact batch shape of a task: (D, N_ctx), plus G in grouped mode."""
    n_groups = t.meta.get("n_groups", 0)
    if n_groups:
        return (t.ctx_x.shape[1], t.ctx_x.shape[0], n_groups)
    return (t.ctx_x.shape[1], t.ctx_x.shape[0])


def num_chunks(num_tasks: int, chunk_size: int) -> int:
    """Number of chunk files needed for `num_tasks` tasks."""
    return math.ceil(num_tasks / chunk_size)
