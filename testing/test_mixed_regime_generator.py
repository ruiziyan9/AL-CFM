"""Tests for CITNP/datasets/mixed_regime_generator.py."""

import dataclasses
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from CITNP.datasets import mixed_regime_generator as mrg
from CITNP.datasets.create_mixed_regime_chunks import main as create_chunks_main
from CITNP.datasets.create_mixed_regime_chunks import write_chunk, write_config
from CITNP.datasets.epoch_callback import SetEpochCallback
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_ADHOC,
    SOURCE_CHUNK,
    SOURCE_STREAM,
    Design,
    MixedRegimeChunkDataset,
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    assert_disjoint_splits,
    collate_tasks,
    descendants,
    generate_task,
    make_key,
    sample_design,
    sample_graph,
    topological_order,
)

FAST_FAMILIES = ["neuralnet", "linearfixedstd", "linearnonidentifiable"]
TASK_ARRAYS = [
    "ctx_x",
    "ctx_int",
    "ctx_regime",
    "qry_x",
    "qry_int",
    "qry_tgt",
    "graph",
]


def make_cfg(family="linearnonidentifiable", **kw) -> MixedRegimeConfig:
    base = dict(
        function_generator=family,
        num_variables=[2, 3, 4, 5],
        graph_type=["ER"],
        graph_degrees={2: [0, 1], 3: [1, 2, 3], 4: [1, 2, 4, 6], 5: [2, 4, 6, 8]},
    )
    base.update(kw)
    return MixedRegimeConfig(**base)  # type: ignore[arg-type]


def chain_graph() -> np.ndarray:
    # 0 -> 1 -> 2, 3 isolated
    g = np.zeros((4, 4), dtype=np.int8)
    g[0, 1] = g[1, 2] = 1
    return g


def obs_do1_design(n=3000, scale=3.0, n_qry=15) -> Design:
    # scale != 1: in a standardised linear-Gaussian SCM, do(X1 ~ N(0, 1)) leaves
    # every marginal unchanged, so the descendant shift would be invisible.
    return Design(
        num_variables=4,
        regime_targets=[(), (1,)],
        regime_sizes=[n, n],
        do_dist={"kind": "normal", "scale": scale},
        qry_targets=[(), (1,), (3,)] * (n_qry // 3),
    )


def assert_tasks_equal(a, b):
    for k in TASK_ARRAYS:
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.regime_targets == b.regime_targets


# 1 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_intervened_cells_exact(family):
    cfg = make_cfg(family)
    for seed in range(10):
        task, dbg = generate_task(cfg, seed, return_debug=True)
        ctx_val = dbg["ctx_int_val"].astype(np.float32)
        qry_val = dbg["qry_int_val"].astype(np.float32)
        assert np.array_equal(task.ctx_x[task.ctx_int], ctx_val[task.ctx_int])
        assert np.array_equal(task.qry_x[task.qry_int], qry_val[task.qry_int])
        # Debug buffers are zero outside intervened cells.
        assert not dbg["ctx_int_val"][~task.ctx_int].any()
        # Same seed regenerates the same do-values.
        assert_tasks_equal(task, generate_task(cfg, seed))


# 2 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_one_call_per_node(family, monkeypatch):
    gen_cls = mrg.REGISTRY[family]
    original = gen_cls.generate_functions
    calls: dict[int, list[int]] = {}

    def wrapped(self, causal_graph):
        fdict = original(self, causal_graph)
        calls.clear()

        def counting(i, f):
            def inner(x):
                calls.setdefault(i, []).append(x.shape[0])
                return f(x)

            return inner

        return {i: counting(i, f) for i, f in fdict.items()}

    monkeypatch.setattr(gen_cls, "generate_functions", wrapped)
    cfg = make_cfg(family)
    for seed in range(10):
        task, dbg = generate_task(cfg, seed, return_debug=True)
        D = task.ctx_x.shape[1]
        assert sorted(calls) == list(range(D))
        for i in range(D):
            assert calls[i] == [dbg["n_tot"]], (i, calls[i])
        assert (
            dbg["n_tot"] == cfg.n_ref + task.ctx_x.shape[0] + task.qry_x.shape[0]
        )


# 3 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_non_descendants_unaffected_descendants_affected(seed):
    cfg = make_cfg("linearfixedstd")
    task = generate_task(cfg, seed, graph=chain_graph(), design=obs_do1_design())
    obs_r = task.regime_targets.index(())
    do_r = next(r for r, t in enumerate(task.regime_targets) if t)
    (target,) = task.regime_targets[do_r]
    desc = descendants(task.graph, target)
    assert len(desc) == 1  # node 2 in the original labelling

    obs = task.ctx_x[task.ctx_regime == obs_r]
    do = task.ctx_x[task.ctx_regime == do_r]
    for v in range(4):
        if v == target:
            continue
        p = ks_2samp(obs[:, v], do[:, v]).pvalue
        if v in desc:
            assert p < 1e-3, (v, p)
        else:
            assert p > 1e-3, (v, p)


# 4 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_reference_standardisation(family):
    # Large n_ref so the reference statistics are accurate well within +-0.1.
    cfg = make_cfg(family, n_ref=8192)
    design = Design(
        num_variables=4,
        regime_targets=[()],
        regime_sizes=[8000],
        do_dist={"kind": "normal", "scale": 1.0},
        qry_targets=[()],
    )
    for seed in range(3):
        task = generate_task(cfg, seed, graph=chain_graph(), design=design)
        assert np.all(np.abs(task.ctx_x.mean(0)) < 0.1)
        std = task.ctx_x.std(0)
        assert np.all((std > 0.9) & (std < 1.1)), std


# 5 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_regime_consistency(family):
    cfg = make_cfg(family)
    for seed in range(20):
        task = generate_task(cfg, seed)
        D = task.ctx_x.shape[1]
        for r, targets in enumerate(task.regime_targets):
            expected = np.zeros(D, dtype=bool)
            expected[list(targets)] = True
            rows = task.ctx_int[task.ctx_regime == r]
            assert len(rows) == task.meta["regime_sizes"][r]
            assert np.all(rows == expected)
        assert np.array_equal(task.qry_tgt, ~task.qry_int)
        assert task.qry_int.sum(1).max(initial=0) <= 1


# 6 ------------------------------------------------------------------------- #
def test_edge_no_obs_regime():
    # Contexts without observational rows only exist with n_obs_min=0 (the
    # trainer's normalisation rejects them; see test_mixed_context_trainer).
    cfg = make_cfg(p_obs=0.0, n_obs_min=0)
    for seed in range(20):
        task = generate_task(cfg, seed)
        assert not task.meta["has_obs"]
        assert task.meta["K"] >= 1
        assert task.ctx_int.any(1).all()


@pytest.mark.parametrize("n_obs_min", [10, 32])
def test_n_obs_min_guarantees_observational_rows(n_obs_min):
    # n_obs_min > 0 overrides p_obs: every task has an observational regime
    # (first) with its one guaranteed row plus n_obs_min reserved rows.
    cfg = make_cfg(p_obs=0.0, n_obs_min=n_obs_min, n_ctx_range=(40, 80))
    for seed in range(20):
        task = generate_task(cfg, seed)
        assert task.meta["has_obs"]
        assert task.regime_targets[0] == ()
        assert task.meta["regime_sizes"][0] >= n_obs_min + 1
        n_obs_rows = int((~task.ctx_int.any(1)).sum())
        assert n_obs_rows == task.meta["regime_sizes"][0]
        assert task.ctx_x.shape[0] == sum(task.meta["regime_sizes"])


def test_n_obs_min_validation():
    with pytest.raises(ValueError, match="n_obs_min"):
        make_cfg(n_obs_min=-1)


def test_edge_k_zero():
    cfg = make_cfg(p_obs=1.0, k_max=0)
    for seed in range(10):
        task = generate_task(cfg, seed)
        assert task.meta["K"] == 0
        assert task.regime_targets == [()]
        assert not task.ctx_int.any()
        assert "seen" not in task.meta["qry_types"]


def test_edge_regime_of_size_one():
    design = Design(
        num_variables=4,
        regime_targets=[(), (2,)],
        regime_sizes=[50, 1],
        do_dist={"kind": "uniform", "width": 4.0},
        qry_targets=[(2,), ()],
    )
    task = generate_task(make_cfg(), 0, design=design)
    assert task.ctx_x.shape[0] == 51
    assert (task.ctx_regime == 1).sum() == 1


def test_edge_all_nodes_targeted():
    cfg = make_cfg(num_variables=[3], p_geom=0.0, k_max=10, n_qry=200)
    n_checked = 0
    for seed in range(40):
        np.random.seed(seed)
        design = sample_design(cfg, 3)
        # p_geom=0 is uniform over K in {0..3}; keep the K=3 draws.
        if sum(1 for t in design.regime_targets if t) != 3:
            continue
        n_checked += 1
        task = generate_task(cfg, seed, design=design)
        assert "unseen" not in task.meta["qry_types"]
        assert set(task.meta["qry_types"]) == {"seen", "empty"}
    assert n_checked > 0


def test_edge_qry_type_renormalisation():
    cfg = make_cfg(qry_type_probs=(1.0, 0.0, 0.0), p_obs=1.0, k_max=0)
    np.random.seed(0)
    design = sample_design(cfg, 3)
    # Only "seen" had weight but none are available -> falls back to empty.
    assert all(t == () for t in design.qry_targets)


def test_edge_two_variables():
    cfg = make_cfg(num_variables=[2])
    for seed in range(20):
        task = generate_task(cfg, seed)
        assert task.ctx_x.shape[1] == 2
        assert task.graph.shape == (2, 2)


# 7 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_reproducibility(family):
    cfg = make_cfg(family)
    a, b = generate_task(cfg, 123), generate_task(cfg, 123)
    assert_tasks_equal(a, b)
    assert a.meta == b.meta
    c = generate_task(cfg, 124)
    assert a.ctx_x.shape != c.ctx_x.shape or not np.array_equal(a.ctx_x, c.ctx_x)


# 8 ------------------------------------------------------------------------- #
def test_collate_contract():
    cfg = make_cfg(n_ctx_range=(15, 40), n_qry=7)
    tasks = [generate_task(cfg, s) for s in range(6)]
    pad_D = 6
    out = collate_tasks(tasks, pad_D=pad_D)
    B = len(tasks)
    Nc = max(t.ctx_x.shape[0] for t in tasks)
    Nq = max(t.qry_x.shape[0] for t in tasks)
    spec = {
        "ctx_x": ((B, Nc, pad_D), torch.float32),
        "ctx_int": ((B, Nc, pad_D), torch.bool),
        "ctx_pad": ((B, Nc), torch.bool),
        "qry_x": ((B, Nq, pad_D), torch.float32),
        "qry_int": ((B, Nq, pad_D), torch.bool),
        "qry_tgt": ((B, Nq, pad_D), torch.bool),
        "qry_pad": ((B, Nq), torch.bool),
        "var_pad": ((B, pad_D), torch.bool),
        "graph": ((B, pad_D, pad_D), torch.int8),
        "ctx_regime": ((B, Nc), torch.int64),
    }
    for k, (shape, dtype) in spec.items():
        assert tuple(out[k].shape) == shape, k
        assert out[k].dtype == dtype, k
    assert len(out["meta"]) == B

    ctx_cell_pad = out["ctx_pad"][:, :, None] | out["var_pad"][:, None, :]
    qry_cell_pad = out["qry_pad"][:, :, None] | out["var_pad"][:, None, :]
    assert (out["ctx_x"][ctx_cell_pad] == 0).all()
    assert (~out["ctx_int"][ctx_cell_pad]).all()
    assert (out["qry_x"][qry_cell_pad] == 0).all()
    assert (~out["qry_int"][qry_cell_pad]).all()
    assert torch.equal(out["qry_tgt"], ~out["qry_int"] & ~qry_cell_pad)
    assert torch.equal(out["ctx_regime"] == -1, out["ctx_pad"])
    for b, t in enumerate(tasks):
        nc, d = t.ctx_x.shape
        assert out["var_pad"][b].tolist() == [False] * d + [True] * (pad_D - d)
        assert np.array_equal(out["ctx_x"][b, :nc, :d].numpy(), t.ctx_x)
        assert np.array_equal(out["graph"][b, :d, :d].numpy(), t.graph)

    # Default pad_D = batch max.
    max_D = max(t.ctx_x.shape[1] for t in tasks)
    assert collate_tasks(tasks)["ctx_x"].shape[2] == max_D


# 9 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAST_FAMILIES)
def test_no_nan_inf(family):
    degrees = {d: list(range(max(1, d // 2), 2 * d + 1)) for d in range(2, 11)}
    cfg = make_cfg(
        family,
        num_variables=list(range(2, 11)),
        graph_type=["ER", "SF"],
        graph_degrees=degrees,
    )
    for seed in range(200):
        task = generate_task(cfg, seed)
        assert np.isfinite(task.ctx_x).all()
        assert np.isfinite(task.qry_x).all()
        assert task.meta["max_abs"] <= cfg.max_abs_value
        assert topological_order(task.graph)  # still a DAG after permutation


# 10 ------------------------------------------------------------------------ #
def test_chunk_round_trip(tmp_path):
    cfg = make_cfg(n_ctx_range=(15, 60), n_qry=8)
    num_tasks, chunk_size, base_seed = 7, 4, 1000
    write_config(cfg, tmp_path, num_tasks, chunk_size, base_seed, split="val")
    for k in range(2):
        write_chunk(
            cfg, tmp_path, k, num_tasks, chunk_size, base_seed, split="val"
        )
    info = json.loads((tmp_path / "config.json").read_text())
    for field in ("device", "torch_version", "numpy_version", "python_version"):
        assert field in info, field
    assert info["split"] == "val" and info["split_id"] == 1

    ds = MixedRegimeChunkDataset(
        str(tmp_path),
        batch_size=3,
        shuffle=False,
        same_variablenum_per_batch=False,
    )
    batches = list(ds)
    direct = [
        generate_task(cfg, make_key(SOURCE_CHUNK, 1, base_seed, i))
        for i in range(num_tasks)
    ]
    expected = [collate_tasks(direct[i : i + 3]) for i in range(0, num_tasks, 3)]
    assert len(batches) == len(expected)
    for got, exp in zip(batches, expected):
        for k in exp:
            if k == "meta":
                assert got[k] == exp[k]
            else:
                assert torch.equal(got[k], exp[k]), k


def test_chunk_dataset_bucketing_and_shuffle(tmp_path):
    cfg = make_cfg(n_ctx_range=(15, 30), n_qry=4)
    for k in range(3):
        write_chunk(cfg, tmp_path, k, 30, 10, 0, split="train")
    ds = MixedRegimeChunkDataset(
        str(tmp_path), batch_size=4, shuffle=True, seed=1, shuffle_buffer_size=16
    )
    seeds = []
    for batch in ds:
        assert not batch["var_pad"].any()  # bucketed by D
        seeds += [m["seed_key"][3] for m in batch["meta"]]
    assert sorted(seeds) == list(range(30))
    assert seeds != list(range(30))


# Datasets / misc ------------------------------------------------------------ #
def test_task_dataset():
    cfg = make_cfg(n_ctx_range=(15, 30), n_qry=4)
    ds = MixedRegimeTaskDataset(
        cfg, batch_size=3, base_seed=5, iterations_per_epoch=4
    )
    batches = list(ds)
    assert len(batches) == 4
    for b in batches:
        assert not b["var_pad"].any()
    # Batch 1 holds counters 3..5 of worker 0, epoch 0, train split.
    assert [m["seed_key"] for m in batches[1]["meta"]] == [
        list(make_key(SOURCE_STREAM, 0, 5, 0, 0, c)) for c in (3, 4, 5)
    ]
    # fresh_each_epoch (default): a second pass continues the counters.
    again = list(ds)
    assert again[0]["meta"][0]["seed_key"] == list(
        make_key(SOURCE_STREAM, 0, 5, 0, 0, 12)
    )


def test_topological_order():
    with pytest.raises(ValueError):
        topological_order(np.array([[0, 1], [1, 0]]))
    g = np.zeros((4, 4), dtype=int)
    g[3, 0] = g[0, 2] = g[2, 1] = 1
    assert topological_order(g) == [3, 0, 2, 1]


def test_task_asdict_roundtrip():
    task = generate_task(make_cfg(), 0)
    assert_tasks_equal(task, mrg.task_from_dict(dataclasses.asdict(task)))


def test_gp_smoke():
    cfg = make_cfg(
        "gp",
        num_variables=[3],
        n_ref=64,
        n_ctx_range=(50, 100),
        n_qry=32,
        max_total_rows_gp=300,
    )
    task = generate_task(cfg, 0)
    assert np.isfinite(task.ctx_x).all()
    big = dataclasses.replace(cfg, n_ctx_range=(400, 400))
    with pytest.raises(ValueError, match="max_total_rows_gp"):
        generate_task(big, 0)


# Fix 2: descendants ---------------------------------------------------------- #
def test_descendants():
    g = chain_graph()  # 0 -> 1 -> 2, 3 isolated
    assert descendants(g, 1) == {2}
    assert descendants(g, 0) == {1, 2}
    assert descendants(g, 3) == set()


# Fix 3: key-based seeding ----------------------------------------------------- #
def test_key_reproducibility():
    cfg = make_cfg()
    key = make_key(SOURCE_CHUNK, 0, 3, 17)
    a, b = generate_task(cfg, key), generate_task(cfg, key)
    assert_tasks_equal(a, b)
    assert a.meta["seed_key"] == list(key)
    assert a.meta["used_key"] == list(key) + [a.meta["resample_attempts"]]
    c = generate_task(cfg, make_key(SOURCE_CHUNK, 0, 3, 18))
    assert a.ctx_x.shape != c.ctx_x.shape or not np.array_equal(a.ctx_x, c.ctx_x)
    # An int seed is shorthand for an ad hoc key.
    assert_tasks_equal(
        generate_task(cfg, 4), generate_task(cfg, make_key(SOURCE_ADHOC, 0, 0, 4))
    )


def _fingerprint(task) -> bytes:
    return task.ctx_x.tobytes() + task.qry_x.tobytes()


def test_splits_differ_at_same_index():
    cfg = make_cfg()
    tr = generate_task(cfg, make_key(SOURCE_CHUNK, 0, 0, 5))
    va = generate_task(cfg, make_key(SOURCE_CHUNK, 1, 0, 5))
    assert _fingerprint(tr) != _fingerprint(va)


def test_adjacent_base_seeds_do_not_overlap():
    # Old scheme: base seeds 0 and 1 shared tasks at shifted indices.
    cfg = make_cfg()
    prints = {
        _fingerprint(generate_task(cfg, make_key(SOURCE_CHUNK, 0, b, i)))
        for b in (0, 1)
        for i in range(10)
    }
    assert len(prints) == 20


@pytest.mark.parametrize(
    "bad",
    [(-1, 0, 0), (0, 0, -5), (0, 0.5, 0), (0, 0, "1")],
)
def test_make_key_validation(bad):
    with pytest.raises(ValueError):
        make_key(*bad)


def test_generate_task_rejects_wrong_key_length():
    with pytest.raises(ValueError):
        generate_task(make_cfg(), (1, 0, 0))


def _write_dir(path, split, base_seed):
    path.mkdir()
    write_config(make_cfg(), path, 1, 1, base_seed, split=split)
    return str(path)


def test_assert_disjoint_splits(tmp_path):
    tr = _write_dir(tmp_path / "train", "train", 0)
    va = _write_dir(tmp_path / "val", "val", 0)
    tr2 = _write_dir(tmp_path / "train2", "train", 1)
    assert_disjoint_splits(tr, va, tr2)  # passes
    dup = _write_dir(tmp_path / "train_dup", "train", 0)
    with pytest.raises(ValueError, match="same tasks"):
        assert_disjoint_splits(tr, va, dup)


# Fix 4: epochs ---------------------------------------------------------------- #
def _stream(split="train", base_seed=0, family="linearnonidentifiable", **kw):
    cfg = make_cfg(family, n_ctx_range=(15, 30), n_qry=4)
    return MixedRegimeTaskDataset(
        cfg,
        batch_size=3,
        base_seed=base_seed,
        iterations_per_epoch=kw.pop("iterations_per_epoch", 2),
        split=split,
        **kw,
    )


def test_stream_not_fresh_repeats():
    ds = _stream(fresh_each_epoch=False)
    a, b = list(ds), list(ds)
    for x, y in zip(a, b):
        assert torch.equal(x["ctx_x"], y["ctx_x"])
        assert x["meta"] == y["meta"]


def test_stream_set_epoch_changes_tasks():
    ds = _stream()
    e0 = list(ds)[0]
    ds.set_epoch(1)
    e1 = list(ds)[0]
    assert e0["ctx_x"].shape != e1["ctx_x"].shape or not torch.equal(
        e0["ctx_x"], e1["ctx_x"]
    )
    keys0 = [m["seed_key"] for m in e0["meta"]]
    keys1 = [m["seed_key"] for m in e1["meta"]]
    assert keys0 != keys1
    assert all(k[3] == 1 for k in keys1)  # epoch is part of the key


def test_stream_train_and_val_differ():
    tr = list(_stream("train"))[0]
    va = list(_stream("val"))[0]
    assert [m["seed_key"][1] for m in va["meta"]] == [1, 1, 1]
    prints_tr = {m["max_abs"] for m in tr["meta"]}
    prints_va = {m["max_abs"] for m in va["meta"]}
    assert prints_tr.isdisjoint(prints_va)


def test_stream_set_epoch_reaches_dataloader_workers():
    ds = _stream(family="linearfixedstd")

    def epoch_batches():
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=2, persistent_workers=False
        )
        return list(dl)

    e0 = epoch_batches()
    assert epoch_batches()[0]["meta"] == e0[0]["meta"]  # same epoch, same tasks
    ds.set_epoch(1)
    e1 = epoch_batches()
    keys0 = {tuple(m["seed_key"]) for b in e0 for m in b["meta"]}
    keys1 = {tuple(m["seed_key"]) for b in e1 for m in b["meta"]}
    assert keys0.isdisjoint(keys1)
    assert {k[4] for k in keys1} == {0, 1}  # both workers contributed


def test_stream_rejects_negative_base_seed():
    with pytest.raises(ValueError):
        _stream(base_seed=-1)


# Fix 6: value bound ----------------------------------------------------------- #
def _small_isolated_case(max_abs_value, attempts):
    # Two isolated roots with N(0, 0.1^2) noise and no standardisation: the
    # largest of ~170 draws is ~0.3, so a 0.28 bound fails on some attempts.
    cfg = make_cfg(
        "linearfixedstd",
        normalise=False,
        n_ref=64,
        max_abs_value=max_abs_value,
        max_resample_attempts=attempts,
    )
    design = Design(2, [()], [20], {"kind": "normal", "scale": 1.0}, [()])
    return cfg, np.zeros((2, 2), dtype=np.int8), design


def test_max_abs_resamples():
    cfg, g, design = _small_isolated_case(0.28, 10)
    attempts = []
    for seed in range(30):
        task = generate_task(cfg, seed, graph=g, design=design)
        assert task.meta["max_abs"] <= 0.28
        assert np.abs(task.ctx_x).max() <= 0.28
        attempts.append(task.meta["resample_attempts"])
    assert max(attempts) > 0  # the bound actually triggered resampling


def test_max_abs_raises_when_attempts_run_out():
    cfg = make_cfg("linearfixedstd", max_abs_value=0.5, max_resample_attempts=0)
    with pytest.raises(RuntimeError, match="max_abs_value"):
        generate_task(cfg, 0)


def test_max_abs_value_validation():
    with pytest.raises(ValueError):
        make_cfg(max_abs_value=0.0)


# Fix 8: graph sampling -------------------------------------------------------- #
def test_sf_small_s0_not_empty():
    cfg = make_cfg(num_variables=[4], graph_type=["SF"], graph_degrees=[1])
    np.random.seed(0)
    g, info = sample_graph(cfg, 4)
    assert info == {"graph_type": "SF", "s0": 4}
    assert g.sum() > 0


def test_graph_meta():
    task = generate_task(make_cfg(graph_type=["ER"]), 0)
    assert task.meta["graph_type"] == "ER"
    assert isinstance(task.meta["s0"], int)
    task = generate_task(make_cfg(), 0, graph=chain_graph())
    assert task.meta["graph_type"] is None and task.meta["s0"] is None
    assert task.meta["device"] in ("cpu", "cuda")


# Part A, Fix 1: fresh tasks under CausalInferenceTrainer ---------------------- #
def _epoch_keys(loader) -> set[tuple[int, ...]]:
    return {tuple(m["seed_key"]) for b in loader for m in b["meta"]}


def _loader(ds, num_workers, persistent):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent,
    )


def test_fresh_epochs_persistent_workers():
    ds = _stream(family="linearfixedstd", iterations_per_epoch=4)
    dl = _loader(ds, 2, True)  # what CausalInferenceTrainer uses
    e1, e2 = _epoch_keys(dl), _epoch_keys(dl)
    assert len(e1) == len(e2) == 12
    assert e1.isdisjoint(e2)


def test_fresh_epochs_no_workers():
    ds = _stream(family="linearfixedstd", iterations_per_epoch=4)
    dl = _loader(ds, 0, False)
    e1, e2 = _epoch_keys(dl), _epoch_keys(dl)
    assert len(e1) == len(e2) == 12
    assert e1.isdisjoint(e2)


def test_fresh_epochs_non_persistent_with_callback():
    ds = _stream(family="linearfixedstd", iterations_per_epoch=4)
    dl = _loader(ds, 2, False)
    cb = SetEpochCallback([ds])
    epochs = []
    for e in range(2):
        cb.on_epoch_begin(None, e)
        epochs.append(_epoch_keys(dl))
    assert len(epochs[0]) == len(epochs[1]) == 12
    assert epochs[0].isdisjoint(epochs[1])


@pytest.mark.parametrize("persistent", [True, False])
def test_not_fresh_repeats_under_workers(persistent):
    ds = _stream(
        family="linearfixedstd", iterations_per_epoch=4, fresh_each_epoch=False
    )
    dl = _loader(ds, 2, persistent)
    a, b = list(dl), list(dl)
    assert len(a) == len(b) == 4
    for x, y in zip(a, b):
        assert x["meta"] == y["meta"]
        assert torch.equal(x["ctx_x"], y["ctx_x"])


def test_set_epoch_callback_calls_every_dataset():
    class Recorder:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    dss = [Recorder(), Recorder()]
    cb = SetEpochCallback(dss)
    cb.on_epoch_begin(None, 3)
    cb.on_epoch_begin(None, 4)
    assert [d.epochs for d in dss] == [[3, 4], [3, 4]]


# Part A, Fix 2: padding-free batches ------------------------------------------ #
# sha256 (first 16 hex chars) of the task arrays for
# make_key(SOURCE_CHUNK, 0, 7, i), i = 0..9, recorded with the code BEFORE the
# n_ctx override was added. Without n_ctx, generation must be byte-identical.
PRE_NCTX_HASHES = {
    "linearfixedstd": [
        "2b75fde5d7738d47", "caab3c21a022fa0e", "075edd4304d7b453",
        "744cf2bec269dd44", "47b5a23611711750", "123da333ff04f453",
        "0f8df0481f2f0544", "3516cb5bb05a23ff", "cc439def5ad844a5",
        "1beaeac8836063bb",
    ],
    "linearnonidentifiable": [
        "d47df259fcb95acc", "b0efa53517b19a36", "085e445b8a98a127",
        "4c758731e3a9f51e", "f1cd5bb216576903", "763d3073591973fa",
        "3c436eeec7e5c9ac", "49c58fe2d3ee7848", "eaf18dda9f60956b",
        "9c1474d16a3f4fa5",
    ],
}  # fmt: skip


def _task_hash(task) -> str:
    h = hashlib.sha256()
    for k in TASK_ARRAYS:
        h.update(getattr(task, k).tobytes())
    h.update(repr(task.regime_targets).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("family", list(PRE_NCTX_HASHES))
def test_generation_unchanged_without_nctx(family):
    cfg = make_cfg(family)
    got = [
        _task_hash(generate_task(cfg, make_key(SOURCE_CHUNK, 0, 7, i)))
        for i in range(10)
    ]
    assert got == PRE_NCTX_HASHES[family]


def test_nctx_override():
    cfg = make_cfg()
    task = generate_task(cfg, make_key(SOURCE_CHUNK, 0, 7, 0), n_ctx=300)
    assert task.ctx_x.shape[0] == 300
    assert sum(task.meta["regime_sizes"]) == 300


def test_nctx_with_design_raises():
    with pytest.raises(ValueError, match="n_ctx"):
        generate_task(make_cfg(), 0, design=obs_do1_design(n=10), n_ctx=300)


def test_nctx_range_lower_bound_validated():
    # k_max=4, max D=5 -> up to 5 regimes, plus n_obs_min (default 10)
    # reserved observational rows, so n_ctx_range[0] must be >= 15.
    with pytest.raises(ValueError, match="n_ctx_range"):
        make_cfg(n_ctx_range=(14, 40))
    make_cfg(n_ctx_range=(15, 40))  # ok
    # Without reserved rows the bound is the number of regimes alone.
    with pytest.raises(ValueError, match="n_ctx_range"):
        make_cfg(n_ctx_range=(4, 40), n_obs_min=0)
    make_cfg(n_ctx_range=(5, 40), n_obs_min=0)  # ok


def _assert_padding_free(batch):
    assert not batch["ctx_pad"].any()
    assert not batch["qry_pad"].any()
    assert not batch["var_pad"].any()


def test_stream_batches_padding_free():
    cfg = make_cfg("linearfixedstd", n_ctx_range=(64, 400), n_qry=16)
    ds = MixedRegimeTaskDataset(
        cfg, batch_size=4, base_seed=0, iterations_per_epoch=20
    )
    batches = list(ds)
    assert len(batches) == 20
    for b in batches:
        _assert_padding_free(b)
    # Batch shapes still vary across batches.
    assert len({tuple(b["ctx_x"].shape) for b in batches}) > 1


def test_chunk_batches_padding_free(tmp_path):
    cfg = make_cfg("linearfixedstd", n_ctx_range=(64, 400), n_qry=16)
    batch_size = 4
    for k in range(2):
        write_chunk(
            cfg,
            tmp_path,
            k,
            80,
            40,
            0,
            split="train",
            shape_group_size=batch_size,
        )
    ds = MixedRegimeChunkDataset(
        str(tmp_path), batch_size=batch_size, shuffle=True, seed=0,
        shuffle_buffer_size=16,
    )  # fmt: skip
    batches = list(ds)
    assert len(batches) == 20
    for b in batches:
        assert b["ctx_x"].shape[0] == batch_size
        _assert_padding_free(b)


def test_grouped_chunks_independent_of_job_split(tmp_path):
    cfg = make_cfg("linearfixedstd", n_ctx_range=(64, 200), n_qry=4)
    args = (24, 8, 3)  # num_tasks, chunk_size, base_seed (multiples of 4)
    one, split = tmp_path / "one", tmp_path / "split"
    one.mkdir()
    split.mkdir()
    for k in range(3):
        write_chunk(cfg, one, k, *args, split="train", shape_group_size=4)
    for k in (2, 0, 1):
        write_chunk(cfg, split, k, *args, split="train", shape_group_size=4)
    for k in range(3):
        a = mrg.load_chunk(str(one / f"chunk_{k:05d}.pt"))
        b = mrg.load_chunk(str(split / f"chunk_{k:05d}.pt"))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tasks_equal(x, y)
    # Every group of 4 consecutive tasks shares one shape.
    tasks = [
        t
        for k in range(3)
        for t in mrg.load_chunk(str(one / f"chunk_{k:05d}.pt"))
    ]
    for g in range(0, 24, 4):
        assert len({t.ctx_x.shape for t in tasks[g : g + 4]}) == 1


def test_cpu_only_device_meta():
    task = generate_task(make_cfg("linearfixedstd"), 0)
    assert task.meta["device"] == "cpu"


# Chunk fixes C1-C3: aligned shape groups and exact len() ---------------------- #
def _chunk_cfg():
    return make_cfg(
        "linearfixedstd", num_variables=[2, 3], n_ctx_range=(16, 64), n_qry=8
    )


def _write_chunks(path, num_tasks, chunk_size, group, base_seed=0):
    cfg = _chunk_cfg()
    path.mkdir(exist_ok=True)
    write_config(
        cfg,
        path,
        num_tasks,
        chunk_size,
        base_seed,
        split="train",
        shape_group_size=group,
    )
    for k in range(-(-num_tasks // chunk_size)):
        write_chunk(
            cfg,
            path,
            k,
            num_tasks,
            chunk_size,
            base_seed,
            split="train",
            shape_group_size=group,
        )
    return str(path)


def test_writer_rejects_misaligned_sizes(tmp_path):
    cfg = _chunk_cfg()
    with pytest.raises(ValueError, match="chunk_size"):
        write_chunk(
            cfg, tmp_path, 0, 200, 100, 0, split="train", shape_group_size=32
        )
    with pytest.raises(ValueError, match="num_tasks"):
        write_chunk(
            cfg, tmp_path, 0, 96, 64, 0, split="train", shape_group_size=64
        )
    assert not list(tmp_path.iterdir())  # nothing was written


@pytest.mark.parametrize(
    "sizes",
    [("200", "100", "32"), ("96", "64", "64")],
)
def test_cli_rejects_misaligned_sizes(sizes, monkeypatch, tmp_path, capsys):
    num_tasks, chunk_size, group = sizes
    argv = [
        "create_mixed_regime_chunks",
        "--config", "unused.yaml",
        "--num_tasks", num_tasks,
        "--chunk_size", chunk_size,
        "--shape_group_size", group,
        "--out_dir", str(tmp_path / "out"),
        "--split", "train",
        "--base_seed", "0",
        "--allow_cuda",  # skip the CUDA guard: this test only checks sizes
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        create_chunks_main()
    assert exc.value.code == 2
    assert "multiple of --shape_group_size" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_groups_stay_inside_chunks(tmp_path):
    d = _write_chunks(tmp_path / "c", 256, 64, 32)
    group_files: dict[int, set[str]] = {}
    for path in mrg.list_chunks(d):
        shapes: dict[int, set] = {}
        for t in mrg.load_chunk(path):
            g = t.meta["seed_key"][3] // 32
            shapes.setdefault(g, set()).add((t.ctx_x.shape[1], t.ctx_x.shape[0]))
            group_files.setdefault(g, set()).add(path)
        assert all(len(v) == 1 for v in shapes.values())
    assert len(group_files) == 256 // 32
    assert all(len(files) == 1 for files in group_files.values())


@pytest.mark.parametrize("num_workers", [0, 2])
def test_chunk_batches_full_and_len_exact(tmp_path, num_workers):
    d = _write_chunks(tmp_path / "c", 256, 64, 32)
    ds = MixedRegimeChunkDataset(d, batch_size=16, shuffle=True, seed=3)
    assert len(ds) == 16
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for _ in range(2):  # two epochs
        batches = list(dl)
        assert len(batches) == len(ds) == 16
        seeds = []
        for b in batches:
            assert b["ctx_x"].shape[0] == 16
            _assert_padding_free(b)
            seeds += [m["seed_key"][3] for m in b["meta"]]
        assert sorted(seeds) == list(range(256))


def test_len_raises_without_shape_groups(tmp_path):
    d = _write_chunks(tmp_path / "c", 64, 32, 0)
    with pytest.warns(UserWarning, match="without shape groups"):
        ds = MixedRegimeChunkDataset(d, batch_size=16)
    with pytest.raises(TypeError, match="not exact"):
        len(ds)


def test_len_raises_when_group_not_multiple_of_batch(tmp_path):
    d = _write_chunks(tmp_path / "c", 256, 64, 32)
    with pytest.warns(UserWarning, match="not a multiple of batch_size"):
        ds = MixedRegimeChunkDataset(d, batch_size=24)
    with pytest.raises(TypeError, match="not exact"):
        len(ds)


def test_len_raises_without_config(tmp_path):
    cfg = _chunk_cfg()
    write_chunk(cfg, tmp_path, 0, 32, 32, 0, split="train", shape_group_size=32)
    ds = MixedRegimeChunkDataset(str(tmp_path), batch_size=16)
    with pytest.raises(TypeError, match="config.json"):
        len(ds)


@pytest.mark.parametrize("drop_last", [False, True])
def test_len_exact_single_bucket_per_worker(tmp_path, drop_last):
    # 100 tasks in chunks of 30 (last chunk 10), no grouping, one bucket.
    d = _write_chunks(tmp_path / "c", 100, 30, 0)
    ds = MixedRegimeChunkDataset(
        d,
        batch_size=8,
        shuffle=True,
        seed=1,
        same_variablenum_per_batch=False,
        drop_last=drop_last,
        num_workers=2,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2)
    assert len(ds) == len(list(dl))


# R1 Part A: grouped query mode ------------------------------------------------ #
# Per-row regression fixtures, recorded with the code BEFORE grouped mode was
# added. They hash only fields and batch keys that existed then.
_FIX_TASK_FIELDS = [
    "ctx_x",
    "ctx_int",
    "ctx_regime",
    "qry_x",
    "qry_int",
    "qry_tgt",
    "graph",
]
_FIX_BATCH_KEYS = [
    "ctx_x", "ctx_int", "ctx_pad", "qry_x", "qry_int", "qry_tgt", "qry_pad",
    "var_pad", "graph", "ctx_regime",
]  # fmt: skip
PRE_GROUP_TASK_HASHES = [
    "a3a52108e0a3d5ad", "3f853072234eadd3", "a1adc7b79d8227f5",
    "2d6373e0ac2f12f6", "caa267d251d4ab66", "7d5dbabf8241fe61",
    "6035e9e92fd12a32", "50f942efaf5ac865", "9d46dbe32058e90b",
    "a628121b40d196e9",
]  # fmt: skip
PRE_GROUP_BATCH_HASHES = [
    "f81c57aac8226d4e",
    "2f77828377a9eca0",
    "6f58afa1cb0a0683",
]
PRE_GROUP_CHUNK_HASH = "09adddc783171b9d"
_FIX_DEGREES = {2: [0, 1], 3: [1, 2, 3], 4: [1, 2, 4, 6], 5: [2, 4, 6, 8]}


def _fix_task_hash(task) -> str:
    h = hashlib.sha256()
    for k in _FIX_TASK_FIELDS:
        h.update(getattr(task, k).tobytes())
    h.update(repr(task.regime_targets).encode())
    return h.hexdigest()[:16]


def _fix_small_cfg():
    return MixedRegimeConfig(
        "linearfixedstd", [2, 3, 4, 5], ["ER"], _FIX_DEGREES,
        n_ctx_range=(16, 64), n_qry=8,
    )  # fmt: skip


def test_per_row_task_hashes_unchanged():
    cfg = MixedRegimeConfig(
        "linearnonidentifiable", [2, 3, 4, 5], ["ER"], _FIX_DEGREES
    )
    got = [
        _fix_task_hash(generate_task(cfg, make_key(SOURCE_CHUNK, 0, 11, i)))
        for i in range(10)
    ]
    assert got == PRE_GROUP_TASK_HASHES


def test_per_row_stream_batch_hashes_unchanged():
    ds = MixedRegimeTaskDataset(
        _fix_small_cfg(),
        batch_size=4,
        base_seed=21,
        iterations_per_epoch=3,
        split="train",
    )
    got = []
    for b in ds:
        h = hashlib.sha256()
        for k in _FIX_BATCH_KEYS:
            h.update(b[k].numpy().tobytes())
        got.append(h.hexdigest()[:16])
        assert (b["qry_group"] == -1).all()
    assert got == PRE_GROUP_BATCH_HASHES


def test_per_row_chunk_hashes_unchanged(tmp_path):
    cfg = _fix_small_cfg()
    for k in range(2):
        write_chunk(
            cfg, tmp_path, k, 32, 16, 5, split="train", shape_group_size=16
        )
    hashes = [
        _fix_task_hash(t)
        for k in range(2)
        for t in mrg.load_chunk(str(tmp_path / f"chunk_{k:05d}.pt"))
    ]
    assert (
        hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
        == PRE_GROUP_CHUNK_HASH
    )


def test_per_row_meta_and_qry_group_defaults():
    task = generate_task(make_cfg(), 0)
    assert task.meta["n_groups"] == 0
    assert task.meta["group_treatments"] == [] and task.meta["group_types"] == []
    assert task.qry_group.dtype == np.int64 and (task.qry_group == -1).all()
    assert collate_tasks([task])["qry_group"].eq(-1).all()


def test_old_format_chunks_still_load(tmp_path):
    # Simulate a chunk written before grouped mode: no qry_group, no new meta.
    cfg = _fix_small_cfg()
    write_config(cfg, tmp_path, 32, 32, 0, split="train", shape_group_size=16)
    write_chunk(cfg, tmp_path, 0, 32, 32, 0, split="train", shape_group_size=16)
    path = tmp_path / "chunk_00000.pt"
    old = torch.load(path, weights_only=False)
    for d in old:
        d.pop("qry_group")
        for k in ("n_groups", "group_treatments", "group_types"):
            d["meta"].pop(k)
    torch.save(old, path)
    ds = MixedRegimeChunkDataset(str(tmp_path), batch_size=16)
    batches = list(ds)
    assert len(batches) == len(ds) == 2
    for b in batches:
        assert (b["qry_group"] == -1).all()
        _assert_padding_free(b)


def _grouped_cfg(**kw):
    base = dict(
        num_variables=[3, 4],
        graph_degrees={3: [1, 2, 3], 4: [2, 3, 4]},
        n_ctx_range=(16, 64),
        n_qry=12,
        qry_mode="grouped",
        qry_groups=[1, 2, 3],
    )
    base.update(kw)
    return make_cfg("linearfixedstd", **base)


@pytest.mark.parametrize("G", [1, 2, 3])
def test_grouped_task_structure(G):
    cfg = _grouped_cfg()
    for seed in range(15):
        task = generate_task(cfg, seed, n_groups=G)
        nq = task.qry_x.shape[0]
        assert nq == cfg.n_qry
        assert np.array_equal(task.qry_group, np.repeat(np.arange(G), nq // G))
        treatments = task.meta["group_treatments"]
        assert task.meta["n_groups"] == G and len(treatments) == G
        assert len(set(treatments)) == G  # distinct across groups
        ctx_cols = set(np.where(task.ctx_int.any(0))[0].tolist())
        for g in range(G):
            rows = task.qry_int[task.qry_group == g]
            assert (rows.sum(1) == 1).all()  # exactly one intervened cell
            cols = set(np.where(rows)[1].tolist())
            assert cols == {treatments[g]}
            expected = "seen" if treatments[g] in ctx_cols else "unseen"
            assert task.meta["group_types"][g] == expected
        assert "empty" not in task.meta["qry_types"]


def test_grouped_sampled_G_and_contiguous():
    cfg = _grouped_cfg()
    Gs = {generate_task(cfg, s).meta["n_groups"] for s in range(30)}
    assert Gs == {1, 2, 3}


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_qry=128, qry_groups=[3]),
        dict(num_variables=[3], graph_degrees={3: [1]}, qry_groups=[4], n_qry=12),
        dict(num_variables=[1], graph_degrees={1: [0]}, qry_groups=[1]),
        dict(qry_type_probs=(0.0, 0.0, 1.0)),
    ],
)
def test_grouped_validation(kw):
    with pytest.raises(ValueError):
        _grouped_cfg(**kw)


def test_per_row_config_ignores_grouped_checks():
    # Settings that would be invalid in grouped mode stay valid in per-row mode.
    make_cfg(n_qry=128, qry_groups=[3])
    with pytest.raises(ValueError, match="qry_mode"):
        make_cfg(qry_mode="per-row")


def test_n_groups_errors():
    with pytest.raises(ValueError, match="grouped"):
        generate_task(make_cfg(), 0, n_groups=2)
    with pytest.raises(ValueError, match="n_groups"):
        generate_task(_grouped_cfg(), 0, design=obs_do1_design(n=10), n_groups=2)


def test_grouped_stream_one_G_per_batch():
    cfg = _grouped_cfg()
    ds = MixedRegimeTaskDataset(
        cfg, batch_size=4, base_seed=0, iterations_per_epoch=20
    )
    Gs = []
    for b in ds:
        _assert_padding_free(b)
        per_task = {m["n_groups"] for m in b["meta"]}
        assert len(per_task) == 1
        G = per_task.pop()
        assert int(b["qry_group"].max()) + 1 == G
        expected = torch.arange(G).repeat_interleave(cfg.n_qry // G)
        assert (b["qry_group"] == expected).all()
        Gs.append(G)
    assert len(set(Gs)) > 1


def test_grouped_chunks_bucket_by_D_Nc_G(tmp_path):
    cfg = _grouped_cfg()
    write_config(cfg, tmp_path, 64, 32, 0, split="train", shape_group_size=8)
    for k in range(2):
        write_chunk(
            cfg, tmp_path, k, 64, 32, 0, split="train", shape_group_size=8
        )
    info = json.loads((tmp_path / "config.json").read_text())
    assert info["config"]["qry_mode"] == "grouped"
    assert info["config"]["qry_groups"] == [1, 2, 3]
    ds = MixedRegimeChunkDataset(
        str(tmp_path), batch_size=8, shuffle=True, seed=0
    )
    batches = list(ds)
    assert len(batches) == len(ds) == 8
    for b in batches:
        _assert_padding_free(b)
        shapes = {(m["D"], b["ctx_x"].shape[1], m["n_groups"]) for m in b["meta"]}
        assert len(shapes) == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_plot_mixed_prior_runs(tmp_path, grouped):
    from CITNP.datasets import plot_mixed_prior

    cfg = _grouped_cfg() if grouped else _fix_small_cfg()
    s = plot_mixed_prior.collect(cfg, 20, 0)
    out = tmp_path / "p.png"
    plot_mixed_prior.plot(s, str(out), "t")
    assert out.exists()


def test_grouped_fresh_epochs_persistent_workers():
    ds = MixedRegimeTaskDataset(
        _grouped_cfg(), batch_size=3, base_seed=0, iterations_per_epoch=4
    )
    dl = _loader(ds, 2, True)
    e1, e2 = _epoch_keys(dl), _epoch_keys(dl)
    assert len(e1) == len(e2) == 12
    assert e1.isdisjoint(e2)
