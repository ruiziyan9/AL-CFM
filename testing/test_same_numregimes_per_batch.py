"""same_numregimes_per_batch: one number of context regimes R = has_obs + K
per batch, without changing the prior over tasks
(CITNP/datasets/mixed_regime_generator.py, create_mixed_regime_chunks.py)."""

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

import CITNP.datasets.create_mixed_regime_chunks as writer
import CITNP.datasets.mixed_regime_generator as generator
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_CHUNK,
    SOURCE_STREAM_D,
    BatchShape,
    MixedRegimeChunkDataset,
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    draw_batch_shape,
    generate_task,
    generate_task_with_shape,
    make_key,
    regime_count_probs,
    sample_design,
)
from CITNP.experiments.mixed_train import build_trainer, parse_args
from CITNP.models.regime_axis import group_regimes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_presimulated_data as presimulated

TASK_ARRAYS = ("ctx_x", "ctx_int", "ctx_regime", "qry_x", "qry_int", "graph")
BATCH_SIZE = 4
SHAPE_GROUP = 8


def make_cfg(**kw) -> MixedRegimeConfig:
    base: dict = dict(
        function_generator="linearfixedstd",
        num_variables=[3],
        graph_type=["ER"],
        graph_degrees={3: [1, 2, 3]},
        k_max=3,
        n_ctx_range=[24, 40],
        n_qry=12,
        qry_mode="grouped",
        qry_groups=[1, 2, 3],
        n_obs_min=8,
    )
    base.update(kw)
    return MixedRegimeConfig(**base)


# Priors with and without n_obs_min, with p_obs < 1, and with p_k0.
PRIORS = {
    "n_obs_min": dict(),
    "p_obs": dict(n_obs_min=0, p_obs=0.6),
    "p_obs_p_k0": dict(n_obs_min=0, p_obs=0.3, p_k0=0.4, p_geom=0.3),
    "n_obs_min_p_k0": dict(p_k0=0.25),
    "k_capped_by_D": dict(n_obs_min=0, p_obs=0.5, k_max=10, p_geom=0.2),
}


def unmodified_regime_count_draw(cfg: MixedRegimeConfig, D: int):
    """The (has_obs, K) draw as it was written inline in `sample_design`
    before the helper existed (verbatim)."""
    k_cap = max(0, min(cfg.k_max, D))
    k_probs = (1.0 - cfg.p_geom) ** np.arange(k_cap + 1)
    if cfg.p_k0 is not None and k_cap > 0:
        k_probs[0] = 0.0
        k_probs = (1.0 - cfg.p_k0) * k_probs / k_probs.sum()
        k_probs[0] = cfg.p_k0
    k_probs = k_probs / k_probs.sum()
    while True:
        has_obs = cfg.n_obs_min > 0 or bool(np.random.rand() < cfg.p_obs)
        K = int(np.random.choice(k_cap + 1, p=k_probs))
        if has_obs or K > 0:
            break
    return has_obs, K


def joint_counts(pairs) -> Counter:
    return Counter((bool(h), int(k)) for h, k in pairs)


def assert_same_distribution(counts_a: Counter, counts_b: Counter, n: int):
    """Two empirical distributions over the same outcomes, n draws each,
    agree within 5 standard errors of the difference."""
    for outcome in set(counts_a) | set(counts_b):
        pa, pb = counts_a[outcome] / n, counts_b[outcome] / n
        p = (pa + pb) / 2
        assert abs(pa - pb) < 5 * np.sqrt(2 * p * (1 - p) / n) + 1e-9, outcome


def num_regimes_per_task(batch) -> torch.Tensor:
    """[B] number of regimes of every task of a collated batch."""
    return group_regimes(batch["ctx_int"]).slot_present.sum(-1)


def write_directory(path, cfg, num_tasks, *, same_numregimes_per_batch):
    kwargs = dict(
        split="train",
        shape_group_size=SHAPE_GROUP,
        same_numregimes_per_batch=same_numregimes_per_batch,
    )
    writer.write_config(cfg, path, num_tasks, num_tasks, 0, **kwargs)
    writer.write_chunk(cfg, path, 0, num_tasks, num_tasks, 0, **kwargs)


# --------------------------------------------------------------------------- #
# Flag off: nothing changes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior", ["n_obs_min", "p_obs_p_k0"])
def test_flag_off_tasks_are_bit_identical(monkeypatch, prior):
    """With no R given, the helper consumes the RNG like the old inline draw."""
    cfg = make_cfg(**PRIORS[prior])
    keys = [make_key(SOURCE_CHUNK, 0, 3, i) for i in range(12)]
    shape = BatchShape(3, 30, 2)
    assert shape.num_regimes is None
    new = [generate_task_with_shape(cfg, key, shape) for key in keys]
    new += [generate_task(cfg, key) for key in keys]

    def old_draw(config, D, num_regimes=None):
        assert num_regimes is None
        return unmodified_regime_count_draw(config, D)

    monkeypatch.setattr(generator, "_draw_regime_counts", old_draw)
    old = [generate_task_with_shape(cfg, key, shape) for key in keys]
    old += [generate_task(cfg, key) for key in keys]
    for a, b in zip(new, old):
        for name in TASK_ARRAYS:
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
        assert a.regime_targets == b.regime_targets
        assert a.meta == b.meta


def test_flag_off_batch_shapes_are_unchanged():
    """R is drawn last: the other fields are the same with and without it."""
    cfg = make_cfg()
    for batch_index in range(30):
        key = list(make_key(SOURCE_STREAM_D, 0, 7, 0, 0, batch_index))
        off = draw_batch_shape(cfg, np.random.default_rng(key))
        on = draw_batch_shape(
            cfg, np.random.default_rng(key), draw_num_regimes=True
        )
        assert off.num_regimes is None and on.num_regimes is not None
        assert on[:3] == off[:3]


def test_streaming_flag_off_is_the_default():
    cfg = make_cfg()
    kwargs = dict(base_seed=0, iterations_per_epoch=2)
    default = list(MixedRegimeTaskDataset(cfg, BATCH_SIZE, **kwargs))
    off = list(
        MixedRegimeTaskDataset(
            cfg, BATCH_SIZE, same_numregimes_per_batch=False, **kwargs
        )
    )
    for a, b in zip(default, off):
        for name in ("ctx_x", "ctx_int", "qry_x", "qry_int"):
            assert torch.equal(a[name], b[name]), name


# --------------------------------------------------------------------------- #
# The prior of R and the conditional draw
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior", list(PRIORS))
def test_regime_count_probs_matches_unmodified_sample_design(prior):
    cfg = make_cfg(**PRIORS[prior])
    D, n = 3, 40_000
    probs = regime_count_probs(cfg, D)
    assert probs.shape == (min(cfg.k_max, D) + 2,)
    assert probs[0] == 0.0 and (probs >= 0).all()
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)
    np.random.seed(0)
    draws = [unmodified_regime_count_draw(cfg, D) for _ in range(n)]
    freq = np.bincount([int(h) + k for h, k in draws], minlength=len(probs)) / n
    tolerance = 5 * np.sqrt(probs * (1 - probs) / n) + 1e-9
    assert (np.abs(freq - probs) < tolerance).all(), (freq, probs)


def test_regime_count_probs_depends_on_num_variables():
    cfg = make_cfg(
        num_variables=[2, 5],
        graph_degrees={2: [1], 5: [2]},
        qry_groups=[1, 2],
        k_max=4,
    )
    assert len(regime_count_probs(cfg, 2)) == 4  # K <= min(k_max, D) = 2
    assert len(regime_count_probs(cfg, 5)) == 6


@pytest.mark.parametrize("prior", list(PRIORS))
def test_prior_is_preserved(prior):
    """R ~ regime_count_probs, then (has_obs, K) | R: the joint distribution
    of (has_obs, K) is the flag-off one."""
    cfg = make_cfg(**PRIORS[prior])
    D, n = 3, 20_000
    rng = np.random.default_rng(1)
    np.random.seed(1)
    on = []
    for _ in range(n):
        R = draw_batch_shape(cfg, rng, draw_num_regimes=True).num_regimes
        design = sample_design(cfg, D, num_regimes=R)
        assert len(design.regime_targets) == R
        on.append(
            (() in design.regime_targets, sum(map(bool, design.regime_targets)))
        )
    np.random.seed(2)
    off = []
    for _ in range(n):
        design = sample_design(cfg, D)
        off.append(
            (() in design.regime_targets, sum(map(bool, design.regime_targets)))
        )
    assert_same_distribution(joint_counts(on), joint_counts(off), n)
    np.random.seed(3)
    unmodified = [unmodified_regime_count_draw(cfg, D) for _ in range(n)]
    assert_same_distribution(joint_counts(on), joint_counts(unmodified), n)


def test_impossible_num_regimes_raises():
    cfg = make_cfg()  # n_obs_min > 0, k_max = 3: R is in 1..4
    np.random.seed(0)
    for R in (0, 5):
        with pytest.raises(ValueError, match="probability zero"):
            sample_design(cfg, 3, num_regimes=R)
    no_obs = make_cfg(n_obs_min=0, p_obs=0.0)  # R = K is in 1..3
    with pytest.raises(ValueError, match="probability zero"):
        sample_design(no_obs, 3, num_regimes=4)
    design = sample_design(no_obs, 3, num_regimes=3)
    assert () not in design.regime_targets


def test_design_and_num_regimes_are_exclusive():
    cfg = make_cfg()
    np.random.seed(0)
    design = sample_design(cfg, 3)
    with pytest.raises(ValueError, match="num_regimes"):
        generate_task(cfg, 0, design=design, num_regimes=2)


# --------------------------------------------------------------------------- #
# Flag on: one R per batch
# --------------------------------------------------------------------------- #


def test_streaming_batches_have_a_single_num_regimes():
    cfg = make_cfg()
    dataset = MixedRegimeTaskDataset(
        cfg,
        BATCH_SIZE,
        base_seed=0,
        iterations_per_epoch=25,
        same_numregimes_per_batch=True,
    )
    seen = Counter()
    for batch in dataset:
        assert group_regimes(batch["ctx_int"]).slot_present.all()
        counts = num_regimes_per_task(batch)
        assert (counts == counts[0]).all()
        # K and has_obs recorded by the generator agree with the data
        for meta, R in zip(batch["meta"], counts.tolist()):
            assert int(meta["has_obs"]) + meta["K"] == R
        seen[int(counts[0])] += 1
    assert len(seen) > 1  # R varies between batches


def test_chunk_batches_have_a_single_num_regimes(tmp_path):
    cfg = make_cfg()
    write_directory(tmp_path, cfg, 48, same_numregimes_per_batch=True)
    info = json.loads((tmp_path / "config.json").read_text())
    assert info["same_numregimes_per_batch"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in particular: no partial-bucket warning
        dataset = MixedRegimeChunkDataset(
            str(tmp_path),
            BATCH_SIZE,
            shuffle=True,
            same_numregimes_per_batch=True,
        )
    batches = list(dataset)
    assert len(batches) == len(dataset) == 48 // BATCH_SIZE  # no partial batch
    seen = set()
    for batch in batches:
        assert batch["ctx_x"].shape[0] == BATCH_SIZE
        assert group_regimes(batch["ctx_int"]).slot_present.all()
        seen.add(int(num_regimes_per_task(batch)[0]))
    assert len(seen) > 1


def test_dataset_level_prior_is_preserved():
    """Tasks of same-R batches follow the flag-off joint of (has_obs, K).
    One R per batch, so the number of batches sets the tolerance."""
    cfg = make_cfg(n_obs_min=0, p_obs=0.6, n_ctx_range=[8, 12])
    batches = 400
    dataset = MixedRegimeTaskDataset(
        cfg, 2, base_seed=5, iterations_per_epoch=batches,
        same_numregimes_per_batch=True,
    )  # fmt: skip
    on = joint_counts(
        (m["has_obs"], m["K"]) for batch in dataset for m in batch["meta"]
    )
    n = 2 * batches
    np.random.seed(0)
    exact = joint_counts(
        unmodified_regime_count_draw(cfg, 3) for _ in range(50 * n)
    )
    for outcome, count in exact.items():
        p = count / (50 * n)
        # standard error with `batches` independent R draws
        assert abs(on[outcome] / n - p) < 5 * np.sqrt(p * (1 - p) / batches)
    assert set(on) <= set(exact)


# --------------------------------------------------------------------------- #
# Streaming and the chunk writer agree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", [False, True])
def test_streaming_and_writer_draw_the_same_batch_shapes(flag):
    cfg = make_cfg(num_variables=[3, 4], graph_degrees={3: [1, 2], 4: [1, 2]})
    for group_index in range(40):
        key = make_key(SOURCE_CHUNK, 1, 5, group_index, 1, 0)
        stream = draw_batch_shape(
            cfg, np.random.default_rng(list(key)), draw_num_regimes=flag
        )
        assert writer.group_shape(cfg, 1, 5, group_index, flag) == tuple(stream)
        assert (stream.num_regimes is not None) == flag


def test_writer_generates_what_streaming_would(tmp_path):
    cfg = make_cfg()
    write_directory(tmp_path, cfg, 16, same_numregimes_per_batch=True)
    stored = generator.load_chunk(str(tmp_path / "chunk_00000.pt"))
    for index, task in enumerate(stored):
        shape = BatchShape(
            *writer.group_shape(cfg, 0, 0, index // SHAPE_GROUP, True)
        )
        direct = generate_task_with_shape(
            cfg, make_key(SOURCE_CHUNK, 0, 0, index), shape
        )
        for name in TASK_ARRAYS:
            assert np.array_equal(getattr(task, name), getattr(direct, name))


# --------------------------------------------------------------------------- #
# Errors and directories written without the flag
# --------------------------------------------------------------------------- #


def test_num_regimes_needs_one_num_variables_per_batch():
    cfg = make_cfg(num_variables=[3, 4], graph_degrees={3: [1, 2], 4: [1, 2]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="one D per batch"):
        draw_batch_shape(
            cfg, rng, draw_num_variables=False, draw_num_regimes=True
        )
    with pytest.raises(ValueError, match="one D per batch"):
        MixedRegimeTaskDataset(
            cfg, BATCH_SIZE, base_seed=0, same_variablenum_per_batch=False,
            same_numregimes_per_batch=True,
        )  # fmt: skip
    # D drawn per batch, or a single D in the config: fine.
    assert draw_batch_shape(cfg, rng, draw_num_regimes=True).num_regimes
    single = make_cfg()
    shape = draw_batch_shape(
        single, rng, draw_num_variables=False, draw_num_regimes=True
    )
    assert shape.num_variables is None and shape.num_regimes


def test_writer_flag_needs_shape_groups(tmp_path):
    cfg = make_cfg()
    with pytest.raises(ValueError, match="shape groups"):
        writer.write_chunk(
            cfg, tmp_path, 0, 8, 8, 0, split="train", shape_group_size=0,
            same_numregimes_per_batch=True,
        )  # fmt: skip


def test_writer_rejects_a_flag_mismatch(tmp_path):
    cfg = make_cfg()
    kwargs = dict(split="train", shape_group_size=SHAPE_GROUP)
    writer.write_config(cfg, tmp_path, 16, 16, 0, **kwargs)
    writer.write_config(cfg, tmp_path, 16, 16, 0, **kwargs)  # identical rerun
    with pytest.raises(ValueError, match="same_numregimes_per_batch"):
        writer.write_config(
            cfg, tmp_path, 16, 16, 0, same_numregimes_per_batch=True, **kwargs
        )
    # A config.json written before the flag existed counts as flag off.
    path = tmp_path / "config.json"
    info = json.loads(path.read_text())
    del info["same_numregimes_per_batch"]
    path.write_text(json.dumps(info))
    writer.write_config(cfg, tmp_path, 16, 16, 0, **kwargs)


def test_directory_written_without_the_flag(tmp_path):
    """Still one R per batch, with a warning about partial buckets."""
    cfg = make_cfg()
    write_directory(tmp_path, cfg, 64, same_numregimes_per_batch=False)
    with pytest.warns(UserWarning, match="partial buckets") as record:
        dataset = MixedRegimeChunkDataset(
            str(tmp_path),
            BATCH_SIZE,
            shuffle=False,
            same_numregimes_per_batch=True,
        )
    assert len([w for w in record if "partial buckets" in str(w.message)]) == 1
    with pytest.raises(TypeError, match="same_numregimes_per_batch"):
        len(dataset)
    batches = list(dataset)
    assert sum(b["ctx_x"].shape[0] for b in batches) == 64  # no task is lost
    sizes = [b["ctx_x"].shape[0] for b in batches]
    assert min(sizes) < BATCH_SIZE  # partial buckets, as warned
    for batch in batches:
        assert group_regimes(batch["ctx_int"]).slot_present.all()
    # With iterations_per_epoch every batch is full: buckets persist.
    with pytest.warns(UserWarning, match="partial buckets"):
        endless = MixedRegimeChunkDataset(
            str(tmp_path), BATCH_SIZE, shuffle=True, iterations_per_epoch=20,
            same_numregimes_per_batch=True,
        )  # fmt: skip
    assert len(endless) == 20
    for batch in endless:
        assert batch["ctx_x"].shape[0] == BATCH_SIZE
        assert group_regimes(batch["ctx_int"]).slot_present.all()
    # Without the option the same directory gives no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MixedRegimeChunkDataset(str(tmp_path), BATCH_SIZE, shuffle=False)


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", [False, True])
def test_mixed_train_passes_the_flag(tmp_path, flag):
    configs = Path(__file__).resolve().parents[1] / "CITNP/experiments/configs"
    args = parse_args(
        [
            "--data_config", str(configs / "smoke_d3.yaml"),
            "--model_config", str(configs / "smoke_d3_model.yaml"),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "2",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--results_path", str(tmp_path), "--run_name", "r",
            *(["--same_numregimes_per_batch"] if flag else []),
        ]
    )  # fmt: skip
    assert args.same_numregimes_per_batch is flag
    trainer, _ = build_trainer(
        args, writer.load_config(configs / "smoke_d3.yaml")
    )
    assert trainer.train_dataset.same_numregimes_per_batch is flag
    assert trainer.validation_dataset.same_numregimes_per_batch is flag
    assert trainer.test_dataset.same_numregimes_per_batch is False
    resolved = yaml.safe_load(
        (tmp_path / "results" / "r" / "resolved_model_config.yaml").read_text()
    )
    assert resolved["same_numregimes_per_batch"] is flag
    if flag:
        for batch in trainer.train_dataset:
            assert group_regimes(batch["ctx_int"]).slot_present.all()


# --------------------------------------------------------------------------- #
# Training from pre-simulated chunks
# --------------------------------------------------------------------------- #


def _presimulated(tmp_path, monkeypatch, *writer_flags):
    """Write the three splits of a tiny dataset; returns the build_trainer
    arguments for training from it with --same_numregimes_per_batch."""
    prior_path = presimulated.write_yaml(
        tmp_path / "tiny_prior.yaml", presimulated.TINY_PRIOR
    )
    dataset_config_path = presimulated.write_yaml(
        tmp_path / "tiny__v1.yaml",
        presimulated.dataset_config_content(prior_path),
    )
    data_root = tmp_path / "data_root"
    for split in ("train", "validation", "test"):
        presimulated.generate_split(
            monkeypatch, dataset_config_path, data_root, split, *writer_flags
        )
    return presimulated.training_args(
        tmp_path, dataset_config_path, data_root, prior_path,
        "--same_numregimes_per_batch", epochs=1,
    )  # fmt: skip


def test_training_from_chunks_written_with_the_flag(tmp_path, monkeypatch):
    args = _presimulated(tmp_path, monkeypatch, "--same_numregimes_per_batch")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no partial-bucket warning
        trainer, _ = build_trainer(
            args, MixedRegimeConfig(**presimulated.TINY_PRIOR)
        )
    assert trainer.train_dataset.same_numregimes_per_batch
    assert trainer.validation_dataset.same_numregimes_per_batch
    assert not trainer.test_dataset.same_numregimes_per_batch
    for dataset in (trainer.train_dataset, trainer.validation_dataset):
        for batch in dataset:
            assert batch["ctx_x"].shape[0] == presimulated.BATCH_SIZE
            assert group_regimes(batch["ctx_int"]).slot_present.all()
    trainer.train()
    assert trainer.global_step == 2


def test_training_from_chunks_written_without_the_flag(tmp_path, monkeypatch):
    """Training chunks would work (bucketed by R, with a warning), but the
    validation split has no exact length: a clear error."""
    args = _presimulated(tmp_path, monkeypatch)
    with pytest.warns(UserWarning, match="partial buckets"):
        with pytest.raises(TypeError, match="--same_numregimes_per_batch"):
            build_trainer(args, MixedRegimeConfig(**presimulated.TINY_PRIOR))
