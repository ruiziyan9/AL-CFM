"""Training from pre-simulated datasets gives the model the inputs that
on-the-fly generation would (CITNP/datasets/dataset_config.py, the writer
create_mixed_regime_chunks.py, MixedRegimeChunkDataset(iterations_per_epoch),
mixed_train --dataset_config). Run on BMRC only."""

import dataclasses
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import CITNP.datasets.create_mixed_regime_chunks as writer
import CITNP.datasets.mixed_regime_generator as generator
from CITNP.datasets.dataset_config import (
    file_fingerprint,
    load_presimulated_dataset_config,
    resolve_data_root,
    split_directory,
)
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_CHUNK,
    SOURCE_STREAM_D,
    BatchShape,
    MixedRegimeChunkDataset,
    MixedRegimeConfig,
    MixedRegimeTaskDataset,
    collate_tasks,
    draw_batch_shape,
    generate_task_with_shape,
    load_chunk,
    make_key,
)
from CITNP.experiments.mixed_train import build_trainer, parse_args
from CITNP.trainer.mixed_context_trainer import prepare_mixed_batch

CONFIGS = Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
TINY_PRIOR = dict(
    function_generator="linear_iw",
    alpha0=1.5,
    num_variables=[3],
    graph_type=["MOTIF3"],
    motif_probs={
        "complete": 0.3,
        "chain": 0.2,
        "fork": 0.2,
        "collider": 0.15,
        "single": 0.1,
        "empty": 0.05,
    },
    graph_degrees=[1],
    normalise=False,
    k_max=3,
    n_ctx_range=[32, 48],
    n_qry=12,
    qry_mode="grouped",
    qry_groups=[1, 2, 3],
    n_obs_min=16,
)
TASKS_PER_SHAPE_GROUP = 8
TASKS_PER_CHUNK_FILE = 16
BATCH_SIZE = 4


def write_yaml(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content))
    return path


@pytest.fixture
def tiny_prior_path(tmp_path) -> Path:
    return write_yaml(tmp_path / "tiny_prior.yaml", TINY_PRIOR)


def dataset_config_content(prior_path: Path, **overrides) -> dict:
    content = dict(
        dataset_name="tiny__v1",
        prior_config=str(prior_path),  # absolute paths are allowed
        tasks_per_chunk_file=TASKS_PER_CHUNK_FILE,
        tasks_per_shape_group=TASKS_PER_SHAPE_GROUP,
        splits={
            "train": {"num_tasks": 64, "base_seed": 0},
            "validation": {"num_tasks": 16, "base_seed": 0},
            "test": {"num_tasks": 16, "base_seed": 0},
        },
    )
    content.update(overrides)
    return content


@pytest.fixture
def dataset_config_path(tmp_path, tiny_prior_path) -> Path:
    return write_yaml(
        tmp_path / "tiny__v1.yaml", dataset_config_content(tiny_prior_path)
    )


def run_writer(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["create_mixed_regime_chunks", *arguments])
    writer.main()


def generate_split(monkeypatch, dataset_config_path, data_root, split, *extra):
    run_writer(
        monkeypatch,
        "--dataset_config", str(dataset_config_path),
        "--split", split,
        "--data_root", str(data_root),
        *extra,
    )  # fmt: skip


@pytest.fixture
def data_root(tmp_path) -> Path:
    return tmp_path / "data_root"


@pytest.fixture
def generated(monkeypatch, dataset_config_path, data_root, tmp_path):
    """All three splits written through the writer's dataset mode."""
    for split in ("train", "validation", "test"):
        generate_split(monkeypatch, dataset_config_path, data_root, split)
    return data_root


# --------------------------------------------------------------------------- #
# 1. Batch shape draws are the old inline draws
# --------------------------------------------------------------------------- #


def old_inline_draw(cfg, key, *, draw_num_variables=True, draw_context_size=True):
    """The draw as it was written inline in the streaming dataset / the writer."""
    rng = np.random.default_rng(list(key))
    D = int(rng.choice(cfg.num_variables)) if draw_num_variables else None
    n_ctx = None
    if draw_context_size:
        lo, hi = cfg.n_ctx_range
        n_ctx = int(rng.integers(lo, hi + 1))
    n_groups = int(rng.choice(cfg.qry_groups)) if cfg.qry_mode == "grouped" else None
    return BatchShape(D, n_ctx, n_groups)


def multi_variable_cfg() -> MixedRegimeConfig:
    return MixedRegimeConfig(
        **{**TINY_PRIOR, "num_variables": [3, 4], "graph_type": ["ER"],
           "motif_probs": None, "graph_degrees": {3: [1, 2], 4: [1, 2]},
           "function_generator": "linearfixedstd", "alpha0": None}
    )  # fmt: skip


@pytest.mark.parametrize("flags", [(True, True), (True, False), (False, True)])
def test_draw_batch_shape_matches_old_stream_draw(flags):
    cfg = multi_variable_cfg()
    for batch_index in range(50):
        key = make_key(SOURCE_STREAM_D, 0, 7, 3, 1, batch_index)
        got = draw_batch_shape(
            cfg,
            np.random.default_rng(list(key)),
            draw_num_variables=flags[0],
            draw_context_size=flags[1],
        )
        assert got == old_inline_draw(
            cfg, key, draw_num_variables=flags[0], draw_context_size=flags[1]
        )


def test_group_shape_matches_old_writer_draw():
    cfg = multi_variable_cfg()
    for group_index in range(50):
        key = make_key(SOURCE_CHUNK, 1, 5, group_index, 1, 0)
        assert writer.group_shape(cfg, 1, 5, group_index) == tuple(
            old_inline_draw(cfg, key)
        )


# --------------------------------------------------------------------------- #
# 2. Both paths call the same generation function with the same kinds of arguments
# --------------------------------------------------------------------------- #


def test_stream_and_writer_use_the_same_generation_call(monkeypatch, tmp_path):
    cfg = MixedRegimeConfig(**TINY_PRIOR)
    calls = {"stream": [], "writer": []}

    def recording(name, module):
        original = module.generate_task_with_shape

        def call(*args, **kwargs):
            calls[name].append((args, kwargs))
            return original(*args, **kwargs)

        return call

    monkeypatch.setattr(
        generator, "generate_task_with_shape", recording("stream", generator)
    )
    monkeypatch.setattr(
        writer, "generate_task_with_shape", recording("writer", writer)
    )
    assert not hasattr(writer, "generate_task"), "the writer must not bypass it"

    stream = MixedRegimeTaskDataset(cfg, 3, base_seed=0, iterations_per_epoch=1)
    next(iter(stream))
    writer.write_chunk(
        cfg, tmp_path, 0, 6, 6, 0, split="train", shape_group_size=3
    )

    assert len(calls["stream"]) == 3 and len(calls["writer"]) == 6

    def signature(call):
        args, kwargs = call
        assert kwargs == {}
        config, key, shape = args
        assert isinstance(config, MixedRegimeConfig)
        assert isinstance(key, tuple) and len(key) == 6
        assert all(isinstance(k, int) for k in key)
        assert isinstance(shape, BatchShape)
        assert all(isinstance(v, int) for v in shape)
        return (type(config), tuple(type(v) for v in shape))

    assert {signature(c) for c in calls["stream"]} == {
        signature(c) for c in calls["writer"]
    }


# --------------------------------------------------------------------------- #
# 3. Round trip: identical prepare_mixed_batch outputs
# --------------------------------------------------------------------------- #


def assert_prepared_batches_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key]), key
        elif key == "norm":
            assert torch.equal(a[key][0], b[key][0]) and torch.equal(
                a[key][1], b[key][1]
            )
        elif key == "meta":
            assert [m["seed_key"] for m in a[key]] == [m["seed_key"] for m in b[key]]
            assert [m["group_treatments"] for m in a[key]] == [
                m["group_treatments"] for m in b[key]
            ]
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("normalise", [True, False])
def test_chunk_round_trip_gives_identical_model_inputs(tmp_path, normalise):
    cfg = MixedRegimeConfig(**TINY_PRIOR)
    base_seed, num_tasks = 3, 16
    writer.write_config(
        cfg, tmp_path, num_tasks, num_tasks, base_seed, split="train",
        shape_group_size=TASKS_PER_SHAPE_GROUP,
    )  # fmt: skip
    writer.write_chunk(
        cfg, tmp_path, 0, num_tasks, num_tasks, base_seed, split="train",
        shape_group_size=TASKS_PER_SHAPE_GROUP,
    )  # fmt: skip

    direct = []
    for index in range(num_tasks):
        shape = BatchShape(
            *writer.group_shape(cfg, 0, base_seed, index // TASKS_PER_SHAPE_GROUP)
        )
        direct.append(
            generate_task_with_shape(
                cfg, make_key(SOURCE_CHUNK, 0, base_seed, index), shape
            )
        )
    direct_batches = [
        collate_tasks(direct[i : i + BATCH_SIZE]) for i in range(0, num_tasks, BATCH_SIZE)
    ]
    dataset = MixedRegimeChunkDataset(str(tmp_path), BATCH_SIZE, shuffle=False)
    chunk_batches = list(dataset)
    assert len(chunk_batches) == len(direct_batches) == num_tasks // BATCH_SIZE

    scale_values = []
    for step, (a, b) in enumerate(zip(direct_batches, chunk_batches)):
        for key in ("ctx_x", "ctx_int", "qry_x", "qry_int", "qry_group", "graph"):
            assert torch.equal(a[key], b[key]), key
        for training in (False, True):  # evaluation draw and training draw
            kwargs = dict(training=training, seed=1, global_step=step, normalise=normalise)
            prepared_direct = prepare_mixed_batch(a, torch.float32, **kwargs)
            prepared_chunk = prepare_mixed_batch(b, torch.float32, **kwargs)
            assert_prepared_batches_equal(prepared_direct, prepared_chunk)
        scale_values.append(prepared_direct["context_scale"])
        assert prepared_direct["group_types"] == prepared_chunk["group_types"]
    scale = torch.cat(scale_values)
    if normalise:
        assert scale.abs().max() > 0, "context_scale should not be trivial"


# --------------------------------------------------------------------------- #
# 4. Same distribution
# --------------------------------------------------------------------------- #


def motif_class(graph: np.ndarray) -> str:
    graph = np.asarray(graph)
    edges = int(graph.sum())
    if edges in (0, 1, 3):
        return {0: "empty", 1: "single", 3: "complete"}[edges]
    if graph.sum(axis=1).max() == 2:
        return "fork"
    if graph.sum(axis=0).max() == 2:
        return "collider"
    return "chain"


def collect_features(items, group: int) -> dict[str, list]:
    """Feature values per name. D, context size and G are shared by the `group`
    consecutive tasks of a shape group (batch), so they are counted once per
    group: only that many independent draws exist."""
    out: dict[str, list] = {}
    for index, (shape, meta, graph) in enumerate(items):
        num_variables, context_size = shape
        per_group = {
            "D": [num_variables],
            "context size (binned)": [context_size // 4],
            "G": [meta["n_groups"]],
        }
        per_task = {
            "K": [meta["K"]],
            "has_obs": [meta["has_obs"]],
            "query types": list(meta["group_types"]),
            "graph class": [motif_class(graph)],
        }
        for name, values in per_task.items():
            out.setdefault(name, []).extend(values)
        if index % group == 0:
            for name, values in per_group.items():
                out.setdefault(name, []).extend(values)
    return out


def test_stream_and_chunk_tasks_have_the_same_distribution(tmp_path):
    cfg = MixedRegimeConfig(**TINY_PRIOR)
    count, group = 2000, 10
    stream = MixedRegimeTaskDataset(
        cfg, group, base_seed=0, iterations_per_epoch=count // group
    )
    stream_items = []
    for batch in stream:
        shape = tuple(batch["ctx_x"].shape[2:0:-1])
        for b in range(group):
            stream_items.append((shape, batch["meta"][b], batch["graph"][b].numpy()))

    writer.write_config(
        cfg, tmp_path, count, count, 0, split="val", shape_group_size=group
    )
    writer.write_chunk(
        cfg, tmp_path, 0, count, count, 0, split="val", shape_group_size=group
    )
    chunk_items = [
        ((t.ctx_x.shape[1], t.ctx_x.shape[0]), t.meta, t.graph)
        for t in load_chunk(str(tmp_path / "chunk_00000.pt"))
    ]
    first = collect_features(stream_items, group)
    second = collect_features(chunk_items, group)
    for name in first:
        a, b = Counter(first[name]), Counter(second[name])
        na, nb = len(first[name]), len(second[name])
        for value in set(a) | set(b):
            pa, pb = a[value] / na, b[value] / nb
            pooled = (a[value] + b[value]) / (na + nb)
            standard_error = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
            assert abs(pa - pb) <= 5 * standard_error + 1e-12, (name, value, pa, pb)


# --------------------------------------------------------------------------- #
# 5. Dataset config
# --------------------------------------------------------------------------- #


def test_dataset_config_loads(dataset_config_path, data_root):
    config = load_presimulated_dataset_config(dataset_config_path)
    assert config.dataset_name == "tiny__v1"
    assert config.splits["train"].num_tasks == 64
    assert split_directory(config, data_root, "validation") == (
        data_root / "tiny__v1" / "validation"
    )


def test_repository_dataset_configs_load():
    for path in sorted((CONFIGS / "datasets").glob("*.yaml")):
        load_presimulated_dataset_config(path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tasks_per_chunk_file": 12}, "multiple of tasks_per_shape_group"),
        ({"tasks_per_shape_group": 0}, "positive integer"),
        ({"splits": {"dev": {"num_tasks": 8, "base_seed": 0}}}, "unknown split"),
        (
            {"splits": {"train": {"num_tasks": 12, "base_seed": 0}}},
            "positive multiple of tasks_per_shape_group",
        ),
        ({"splits": {"train": {"num_tasks": 8}}}, "num_tasks and base_seed"),
        ({"dataset_name": "a/b"}, "plain directory name"),
        ({"prior_config": "does/not/exist.yaml"}, "not found"),
    ],
)
def test_dataset_config_rejects(tmp_path, tiny_prior_path, overrides, message):
    path = write_yaml(
        tmp_path / "bad.yaml", dataset_config_content(tiny_prior_path, **overrides)
    )
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_presimulated_dataset_config(path)


def test_dataset_config_rejects_per_row_prior(tmp_path):
    per_row = {k: v for k, v in TINY_PRIOR.items() if k != "qry_groups"}
    prior = write_yaml(tmp_path / "per_row.yaml", {**per_row, "qry_mode": "per_row"})
    path = write_yaml(tmp_path / "d.yaml", dataset_config_content(prior))
    with pytest.raises(ValueError, match="grouped"):
        load_presimulated_dataset_config(path)


def test_data_root_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("ALCFM_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="ALCFM_DATA_ROOT"):
        resolve_data_root(None)
    monkeypatch.setenv("ALCFM_DATA_ROOT", str(tmp_path / "from_environment"))
    assert resolve_data_root(None) == (tmp_path / "from_environment").resolve()
    assert resolve_data_root(str(tmp_path / "given")) == (tmp_path / "given").resolve()


def test_data_root_inside_the_repository_is_refused():
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside the git working tree"):
        resolve_data_root(str(repository / "data"))


# --------------------------------------------------------------------------- #
# 6. Writer
# --------------------------------------------------------------------------- #


def test_writer_records_provenance(generated):
    info = json.loads((generated / "tiny__v1" / "train" / "config.json").read_text())
    assert info["dataset_name"] == "tiny__v1"
    assert info["prior_config_path"].endswith("tiny_prior.yaml")
    assert info["prior_config_fingerprint"] == file_fingerprint(info["prior_config_path"])
    assert isinstance(info["git_has_uncommitted_changes"], (bool, type(None)))
    assert info["generation_procedure"] == "generate_task_with_shape"
    assert info["split"] == "train" and info["shape_group_size"] == TASKS_PER_SHAPE_GROUP
    val = json.loads((generated / "tiny__v1" / "validation" / "config.json").read_text())
    assert val["split"] == "val"  # "validation" is "val" inside the generator
    assert len(list((generated / "tiny__v1" / "train").glob("chunk_*.pt"))) == 4


def test_writer_guard(monkeypatch, tmp_path, tiny_prior_path, dataset_config_path, data_root):
    generate_split(monkeypatch, dataset_config_path, data_root, "validation")
    generate_split(monkeypatch, dataset_config_path, data_root, "validation")  # identical rerun

    changed = write_yaml(
        tmp_path / "changed.yaml",
        dataset_config_content(
            tiny_prior_path,
            splits={"validation": {"num_tasks": 32, "base_seed": 0}},
        ),
    )
    with pytest.raises(ValueError, match="num_tasks"):
        generate_split(monkeypatch, changed, data_root, "validation")
    with pytest.raises(ValueError, match="different generation settings"):
        writer.write_config(
            MixedRegimeConfig(**{**TINY_PRIOR, "n_qry": 24}),
            data_root / "tiny__v1" / "validation", 16, 16, 0, split="val",
            shape_group_size=TASKS_PER_SHAPE_GROUP,
        )  # fmt: skip


def test_skip_existing_files(monkeypatch, dataset_config_path, data_root):
    directory = data_root / "tiny__v1" / "train"
    generate_split(monkeypatch, dataset_config_path, data_root, "train", "--chunk_index", "1")
    existing = directory / "chunk_00001.pt"
    before = (existing.read_bytes(), existing.stat().st_mtime_ns)
    generate_split(monkeypatch, dataset_config_path, data_root, "train", "--skip_existing_files")
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before
    assert len(list(directory.glob("chunk_*.pt"))) == 4
    (directory / "chunk_00002.pt").write_bytes(b"damaged")  # does not load: rewritten
    generate_split(monkeypatch, dataset_config_path, data_root, "train", "--skip_existing_files")
    assert len(load_chunk(str(directory / "chunk_00002.pt"))) == TASKS_PER_CHUNK_FILE


def test_job_index_output_is_byte_identical_to_one_process(
    monkeypatch, tmp_path, dataset_config_path
):
    one, split_over_jobs = tmp_path / "one", tmp_path / "jobs"
    generate_split(monkeypatch, dataset_config_path, one, "train")
    for job_index in range(3):  # 4 files over 3 jobs: job 0 writes files 0 and 3
        generate_split(
            monkeypatch, dataset_config_path, split_over_jobs, "train",
            "--job_index", str(job_index), "--num_jobs", "3",
        )  # fmt: skip
    names = sorted(p.name for p in (one / "tiny__v1" / "train").glob("chunk_*.pt"))
    assert len(names) == 4
    for name in names:
        assert (one / "tiny__v1" / "train" / name).read_bytes() == (
            split_over_jobs / "tiny__v1" / "train" / name
        ).read_bytes()


def test_dataset_mode_is_exclusive_with_direct_flags(monkeypatch, dataset_config_path, data_root):
    with pytest.raises(SystemExit):
        generate_split(monkeypatch, dataset_config_path, data_root, "train", "--num_tasks", "8")


# --------------------------------------------------------------------------- #
# 7. iterations_per_epoch
# --------------------------------------------------------------------------- #


def collect_epochs(loader, epochs):
    return [list(loader) for _ in range(epochs)]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_iterations_per_epoch_across_a_pass_boundary(generated, num_workers):
    directory = str(generated / "tiny__v1" / "train")  # 64 tasks = 16 batches per pass
    dataset = MixedRegimeChunkDataset(
        directory, BATCH_SIZE, shuffle=True, seed=0,
        iterations_per_epoch=6, num_workers=num_workers,
    )  # fmt: skip
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )  # fmt: skip
    assert len(loader) == 6
    epochs = collect_epochs(loader, 3)  # 18 batches: the pass ends after 16
    keys = []
    for batches in epochs:
        assert len(batches) == 6
        for batch in batches:
            assert batch["ctx_x"].shape[0] == BATCH_SIZE
            assert not batch["ctx_pad"].any() and not batch["var_pad"].any()
            keys.extend(tuple(m["seed_key"]) for m in batch["meta"])
    first_pass, second_pass = keys[: 16 * BATCH_SIZE], keys[16 * BATCH_SIZE :]
    assert len(set(first_pass)) == 64, "a task repeated inside the first pass"
    assert len(set(second_pass)) == len(second_pass) == 2 * BATCH_SIZE


def test_iterations_per_epoch_none_keeps_one_pass_per_epoch(generated):
    directory = str(generated / "tiny__v1" / "train")
    dataset = MixedRegimeChunkDataset(directory, BATCH_SIZE, shuffle=True, seed=0)
    assert len(dataset) == 16 and len(list(dataset)) == 16


def test_iterations_per_epoch_rejects_non_positive(generated):
    with pytest.raises(ValueError):
        MixedRegimeChunkDataset(
            str(generated / "tiny__v1" / "train"), BATCH_SIZE, iterations_per_epoch=0
        )


# --------------------------------------------------------------------------- #
# 8. + 9. Training
# --------------------------------------------------------------------------- #


def training_args(tmp_path, dataset_config_path, data_root, prior_path, *extra, epochs=2):
    return parse_args(
        [
            "--data_config", str(prior_path),
            "--dataset_config", str(dataset_config_path),
            "--data_root", str(data_root),
            "--model_config", str(CONFIGS / "smoke_d3_model.yaml"),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", str(BATCH_SIZE), "--iterations_per_epoch", "2",
            "--epochs", str(epochs), "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--save_checkpoint_every_n_steps", "1000",
            "--results_path", str(tmp_path / "results_root"), "--run_name", "run", *extra,
        ]
    )  # fmt: skip


@pytest.mark.parametrize("encoding", ["r1", "layerwise", "extra_node"])
@pytest.mark.parametrize("scale_conditioning", ["none", "stats"])
def test_training_from_chunks_runs(
    tmp_path, generated, dataset_config_path, tiny_prior_path, encoding, scale_conditioning
):
    args = training_args(
        tmp_path, dataset_config_path, generated, tiny_prior_path,
        "--intervention_encoding", encoding,
        "--scale_conditioning", scale_conditioning,
    )  # fmt: skip
    trainer, _ = build_trainer(args, MixedRegimeConfig(**TINY_PRIOR))
    assert isinstance(trainer.train_dataset, MixedRegimeChunkDataset)
    assert isinstance(trainer.validation_dataset, MixedRegimeChunkDataset)
    assert isinstance(trainer.test_dataset, MixedRegimeChunkDataset)
    trainer.train()
    assert trainer.global_step == 4
    stats = trainer.test()
    assert math.isfinite(stats["test_loss_all_mean"])
    run_directory = tmp_path / "results_root" / "results" / "run"
    record = yaml.safe_load((run_directory / "presimulated_data.yaml").read_text())
    assert set(record["splits"]) == {"train", "validation", "test"}


def test_prior_mismatch_raises(tmp_path, generated, dataset_config_path, tiny_prior_path):
    other = write_yaml(tmp_path / "other.yaml", {**TINY_PRIOR, "n_qry": 24})
    args = training_args(tmp_path, dataset_config_path, generated, other)
    with pytest.raises(ValueError, match="differs from the prior"):
        build_trainer(args, MixedRegimeConfig(**{**TINY_PRIOR, "n_qry": 24}))

    # chunks written from another prior than the dataset config names
    stored_path = generated / "tiny__v1" / "train" / "config.json"
    stored = json.loads(stored_path.read_text())
    stored["config"]["n_qry"] = 24
    stored_path.write_text(json.dumps(stored))
    args = training_args(tmp_path, dataset_config_path, generated, tiny_prior_path)
    with pytest.raises(ValueError, match="different prior"):
        build_trainer(args, MixedRegimeConfig(**TINY_PRIOR))


def test_missing_evaluation_splits_fall_back_to_streaming(
    monkeypatch, tmp_path, dataset_config_path, tiny_prior_path, data_root
):
    generate_split(monkeypatch, dataset_config_path, data_root, "train")
    args = training_args(tmp_path, dataset_config_path, data_root, tiny_prior_path)
    trainer, _ = build_trainer(args, MixedRegimeConfig(**TINY_PRIOR))
    assert isinstance(trainer.train_dataset, MixedRegimeChunkDataset)
    assert isinstance(trainer.validation_dataset, MixedRegimeTaskDataset)
    assert isinstance(trainer.test_dataset, MixedRegimeTaskDataset)
    trainer.train()


def test_missing_training_split_raises(
    tmp_path, dataset_config_path, tiny_prior_path, data_root
):
    args = training_args(tmp_path, dataset_config_path, data_root, tiny_prior_path)
    with pytest.raises(FileNotFoundError, match="No simulated training data"):
        build_trainer(args, MixedRegimeConfig(**TINY_PRIOR))


def test_resume_full_in_chunk_mode(tmp_path, generated, dataset_config_path, tiny_prior_path):
    cfg = MixedRegimeConfig(**TINY_PRIOR)
    trainer, _ = build_trainer(
        training_args(tmp_path, dataset_config_path, generated, tiny_prior_path, epochs=1), cfg
    )
    trainer.train()
    checkpoint = tmp_path / "results_root" / "results" / "run" / "last_model.pt"
    resumed, _ = build_trainer(
        training_args(
            tmp_path, dataset_config_path, generated, tiny_prior_path,
            "--resume_full", str(checkpoint), epochs=2,
        ),
        cfg,
    )  # fmt: skip
    assert resumed.global_step == 2 and resumed.current_epoch == 1
    resumed.train()
    assert resumed.global_step == 4


def test_dataclass_round_trip_of_the_config_is_json_stable():
    """The comparison in mixed_train relies on this."""
    cfg = MixedRegimeConfig(**TINY_PRIOR)
    once = json.loads(json.dumps(dataclasses.asdict(cfg)))
    assert json.loads(json.dumps(once)) == once
