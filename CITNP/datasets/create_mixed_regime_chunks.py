"""
Write mixed-regime tasks to chunk files (unpadded; collation happens at load).

Task key = make_key(SOURCE_CHUNK, split_id, base_seed, global_task_index), so
the output does not depend on how the chunks are split across jobs, and
train / val / test (different --split) never share tasks.

Consecutive groups of --shape_group_size tasks (default 256) share one
(D, N_ctx), so MixedRegimeChunkDataset can form padding-free batches. Groups
are keyed by global group index, so this is also independent of the split
into jobs.

chunk_size and num_tasks must be multiples of shape_group_size; shape_group_size
should be a multiple of the training batch size.

Runs on CPU by default, so the same key gives the same task on every node
(CPU and CUDA random streams differ). Pass --allow_cuda to let GP/NN
mechanisms use the GPU.

Example (all chunks in one process):
    python -m CITNP.datasets.create_mixed_regime_chunks \
        --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
        --num_tasks 102400 --chunk_size 1024 --out_dir data/mixed_lg/train \
        --split train --base_seed 0 --shape_group_size 256

Pre-simulated dataset mode: the settings come from a dataset config
(CITNP/datasets/dataset_config.py) and the files go to
<data_root>/<dataset_name>/<split>. This is mutually exclusive with --config,
--num_tasks, --chunk_size, --out_dir, --base_seed and --shape_group_size:

    python -m CITNP.datasets.create_mixed_regime_chunks \
        --dataset_config CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml \
        --split validation --data_root $ALCFM_DATA_ROOT

    # On BMRC, a fixed-size CPU array covers any number of chunk files: job i of
    # n writes chunk files i, i + n, i + 2n, ...; files that already exist and
    # load are skipped, so a resubmitted array only fills the gaps.
    DATASET_CONFIG=CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml \
    SPLIT=train sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh

Example SLURM array on Oxford ARC (one chunk per array job):

    #!/bin/bash
    #SBATCH --job-name=mixed_chunks
    #SBATCH --partition=short
    #SBATCH --time=02:00:00
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    #SBATCH --array=0-99            # = num_tasks / chunk_size - 1
    #SBATCH --output=logs/mixed_chunks_%A_%a.out

    module load Anaconda3
    source activate $DATA/envs/citnp
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

    python -m CITNP.datasets.create_mixed_regime_chunks \
        --config CITNP/datasets/configs/mixed_linear_gaussian.yaml \
        --num_tasks 102400 --chunk_size 1024 \
        --out_dir $DATA/mixed_lg/train --split train --base_seed 0 \
        --shape_group_size 256 --chunk_index $SLURM_ARRAY_TASK_ID
"""

import os
import sys

# Force CPU before torch initialises CUDA. Only when run as a script: other
# modules (CITNP.prior_diagnostics, tests) import load_config from here and must keep
# their CUDA setting. The mechanisms in functions_generator.py choose their
# device when they are constructed, so this is early enough.
if __name__ == "__main__" and "--allow_cuda" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import dataclasses
import io
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from CITNP.datasets.dataset_config import (
    file_fingerprint,
    generator_split_name,
    load_presimulated_dataset_config,
    prior_config_path,
    resolve_data_root,
    split_directory,
)
from CITNP.datasets.mixed_regime_generator import (
    SOURCE_CHUNK,
    SPLIT_IDS,
    BatchShape,
    MixedRegimeConfig,
    draw_batch_shape,
    generate_task_with_shape,
    make_key,
    num_chunks,
)

GENERATION_PROCEDURE = "generate_task_with_shape"
# config.json entries that decide which tasks a directory holds; a directory
# is never filled with tasks from different settings.
GENERATION_SETTINGS = (
    "config",
    "num_tasks",
    "chunk_size",
    "base_seed",
    "split",
    "shape_group_size",
)


def load_config(path: str | Path) -> MixedRegimeConfig:
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]

            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    return MixedRegimeConfig(**raw)


def _git_has_uncommitted_changes() -> bool | None:
    """True if the working tree differs from HEAD; None if git is unavailable."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return None


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _atomic_write(path: Path, write_fn) -> None:
    # Concurrent array jobs may write the same file (config.json).
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def write_config(
    cfg: MixedRegimeConfig,
    out_dir: Path,
    num_tasks: int,
    chunk_size: int,
    base_seed: int,
    *,
    split: str,
    shape_group_size: int = 0,
    dataset_name: str | None = None,
    prior_config_file: str | Path | None = None,
) -> None:
    """Write config.json: the config, split and seed, git commit and runtime.

    If out_dir already has a config.json, its generation settings
    (GENERATION_SETTINGS) must equal the new ones, otherwise ValueError: tasks
    of different settings are never mixed in one directory. A different git
    commit only warns. `prior_config_file` is the file `cfg` was loaded from;
    its path and fingerprint are recorded.
    """
    info = {
        "dataset_name": dataset_name,
        "prior_config_path": (
            None if prior_config_file is None else str(prior_config_file)
        ),
        "prior_config_fingerprint": (
            None if prior_config_file is None else file_fingerprint(prior_config_file)
        ),
        "generation_procedure": GENERATION_PROCEDURE,
        "config": dataclasses.asdict(cfg),
        "num_tasks": num_tasks,
        "chunk_size": chunk_size,
        "base_seed": base_seed,
        "split": split,
        "split_id": SPLIT_IDS[split],
        "shape_group_size": shape_group_size,
        "git_commit": _git_commit(),
        "git_has_uncommitted_changes": _git_has_uncommitted_changes(),
        # Tasks are reproducible only on the same device type and versions.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }

    _check_matches_existing(out_dir / "config.json", info)

    def _write(p):
        with open(p, "w") as f:
            json.dump(info, f, indent=2)

    _atomic_write(out_dir / "config.json", _write)


def _check_matches_existing(path: Path, info: dict) -> None:
    """Raise ValueError if `path` exists with other generation settings."""
    if not path.exists():
        return
    with open(path) as f:
        existing = json.load(f)
    # Compare as JSON, so tuples and lists agree.
    new = json.loads(json.dumps(info))
    different = [k for k in GENERATION_SETTINGS if existing.get(k) != new[k]]
    if different:
        raise ValueError(
            f"{path} was written with different generation settings "
            f"({', '.join(different)}); use another directory instead of "
            "mixing two datasets in one"
        )
    if existing.get("git_commit") != new["git_commit"]:
        warnings.warn(
            f"{path} was written at git commit {existing.get('git_commit')}, "
            f"now {new['git_commit']}; the settings are identical",
            stacklevel=2,
        )


def group_shape(
    cfg: MixedRegimeConfig, split_id: int, base_seed: int, group_index: int
) -> tuple[int, int, int | None]:
    """(D, n_ctx, n_groups) shared by all tasks of one shape group.

    Drawn from its own key (i2 = 1 keeps it apart from task keys, whose i2 is
    0) by draw_batch_shape, the same draw the streaming dataset makes per batch.
    n_groups is None in per-row mode.
    """
    rng = np.random.default_rng(
        list(make_key(SOURCE_CHUNK, split_id, base_seed, group_index, 1, 0))
    )
    shape = draw_batch_shape(cfg, rng)
    assert shape.num_variables is not None and shape.context_size is not None
    return shape.num_variables, shape.context_size, shape.num_question_groups


def check_group_alignment(
    num_tasks: int, chunk_size: int, shape_group_size: int
) -> None:
    """Raise ValueError if shape groups would straddle chunks or be incomplete."""
    if shape_group_size <= 0:
        return
    if chunk_size % shape_group_size != 0:
        raise ValueError(
            "chunk_size must be a multiple of shape_group_size (otherwise shape "
            "groups straddle chunk files and are split across DataLoader workers)"
        )
    if num_tasks % shape_group_size != 0:
        raise ValueError(
            "num_tasks must be a multiple of shape_group_size (otherwise the "
            "last shape group is incomplete)"
        )


def write_chunk(
    cfg: MixedRegimeConfig,
    out_dir: Path,
    chunk_index: int,
    num_tasks: int,
    chunk_size: int,
    base_seed: int,
    *,
    split: str,
    shape_group_size: int = 0,
    show_progress: bool = False,
) -> Path:
    """Generate tasks [chunk_index * chunk_size, ...) and save them as one chunk.

    With shape_group_size G > 0, tasks with the same global_task_index // G
    share one (D, n_ctx), drawn from the group's own key; G = 0 samples the
    shape per task. chunk_size and num_tasks must then be multiples of G, so
    no group straddles two chunk files (files are sharded across DataLoader
    workers, which would split the group).
    """
    check_group_alignment(num_tasks, chunk_size, shape_group_size)
    split_id = SPLIT_IDS[split]
    start = chunk_index * chunk_size
    end = min(start + chunk_size, num_tasks)
    if start >= end:
        raise ValueError(f"chunk_index {chunk_index} is out of range")
    tasks = []
    it = trange(start, end) if show_progress else range(start, end)
    for idx in it:
        key = make_key(SOURCE_CHUNK, split_id, base_seed, idx)
        if shape_group_size > 0:
            shape = BatchShape(
                *group_shape(cfg, split_id, base_seed, idx // shape_group_size)
            )
        else:
            shape = BatchShape(None, None, None)
        task = generate_task_with_shape(cfg, key, shape)
        tasks.append(dataclasses.asdict(task))
    path = out_dir / f"chunk_{chunk_index:05d}.pt"

    def _save(p):
        # Serialise in memory first: torch.save into a named file stores the
        # file name inside the archive, which would make the bytes depend on the
        # temporary name.
        buffer = io.BytesIO()
        torch.save(tasks, buffer)
        p.write_bytes(buffer.getvalue())

    _atomic_write(path, _save)
    return path


def chunk_file_is_complete(path: Path, expected_num_tasks: int) -> bool:
    """True if `path` exists and loads as a list of `expected_num_tasks` tasks.

    Writes are atomic (a temporary file is renamed at the end), so an existing
    file is normally complete; loading also catches a damaged one.
    """
    if not path.exists():
        return False
    try:
        return len(torch.load(path, weights_only=False)) == expected_num_tasks
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    direct = parser.add_argument_group(
        "direct mode", "settings given on the command line"
    )
    direct.add_argument("--config", help="yaml or json config")
    direct.add_argument("--num_tasks", type=int)
    direct.add_argument("--chunk_size", type=int)
    direct.add_argument("--out_dir")
    direct.add_argument("--base_seed", type=int)
    direct.add_argument(
        "--shape_group_size",
        type=int,
        default=None,
        help="Consecutive tasks sharing one (D, N_ctx) for padding-free "
        "batches; use a multiple of the batch size. 0 disables grouping. "
        "Default 256.",
    )
    dataset = parser.add_argument_group(
        "pre-simulated dataset mode",
        "settings from a dataset config; exclusive with the direct-mode options",
    )
    dataset.add_argument("--dataset_config", help="dataset config yaml")
    dataset.add_argument(
        "--data_root",
        default=None,
        help="Directory of simulated datasets (default: environment variable "
        "ALCFM_DATA_ROOT); must be outside the git working tree",
    )
    parser.add_argument(
        "--split",
        required=True,
        help=f"Data split; train / val / test never share tasks. Direct mode: "
        f"{list(SPLIT_IDS)}. Dataset mode: train, validation or test.",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=None,
        help="Write only this chunk (e.g. $SLURM_ARRAY_TASK_ID); default: all",
    )
    parser.add_argument(
        "--job_index",
        type=int,
        default=None,
        help="With --num_jobs: write chunk files job_index, job_index + "
        "num_jobs, job_index + 2 * num_jobs, ... so a fixed-size SLURM array "
        "covers any number of files",
    )
    parser.add_argument("--num_jobs", type=int, default=None)
    parser.add_argument(
        "--skip_existing_files",
        action="store_true",
        help="Skip chunk files that already exist and load (writes are atomic, "
        "so an existing file is complete); makes a resubmitted array safe",
    )
    parser.add_argument(
        "--allow_cuda",
        action="store_true",
        help="Let GP/NN mechanisms use CUDA. Tasks are then reproducible only "
        "on the same device type.",
    )
    args = parser.parse_args()
    if not args.allow_cuda and torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is visible although --allow_cuda was not set: something "
            "initialised CUDA before CUDA_VISIBLE_DEVICES was set"
        )
    if (args.job_index is None) != (args.num_jobs is None):
        parser.error("--job_index and --num_jobs go together")
    if args.job_index is not None:
        if args.chunk_index is not None:
            parser.error("--chunk_index and --job_index are exclusive")
        if args.num_jobs < 1 or not 0 <= args.job_index < args.num_jobs:
            parser.error("need num_jobs >= 1 and 0 <= job_index < num_jobs")

    prior_config_file = None
    dataset_name = None
    if args.dataset_config is not None:
        given = [
            f"--{name}"
            for name in (
                "config",
                "num_tasks",
                "chunk_size",
                "out_dir",
                "base_seed",
                "shape_group_size",
            )
            if getattr(args, name) is not None
        ]
        if given:
            parser.error(f"--dataset_config is exclusive with {', '.join(given)}")
        dataset_config = load_presimulated_dataset_config(args.dataset_config)
        if args.split not in dataset_config.splits:
            parser.error(
                f"split {args.split!r} is not in {args.dataset_config} "
                f"(has {list(dataset_config.splits)})"
            )
        settings = dataset_config.splits[args.split]
        prior_config_file = prior_config_path(dataset_config)
        dataset_name = dataset_config.dataset_name
        cfg = load_config(prior_config_file)
        out_dir = split_directory(
            dataset_config, resolve_data_root(args.data_root), args.split
        )
        split = generator_split_name(args.split)
        num_tasks = settings.num_tasks
        chunk_size = dataset_config.tasks_per_chunk_file
        base_seed = settings.base_seed
        shape_group_size = dataset_config.tasks_per_shape_group
    else:
        missing = [
            f"--{name}"
            for name in ("config", "num_tasks", "chunk_size", "out_dir", "base_seed")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required without --dataset_config")
        if args.data_root is not None:
            parser.error("--data_root needs --dataset_config")
        if args.split not in SPLIT_IDS:
            parser.error(f"--split must be one of {list(SPLIT_IDS)}")
        if args.base_seed < 0:
            parser.error("--base_seed must be >= 0")
        shape_group_size = 256 if args.shape_group_size is None else args.shape_group_size
        if shape_group_size < 0:
            parser.error("--shape_group_size must be >= 0")
        if shape_group_size > 0:
            if args.chunk_size % shape_group_size != 0:
                parser.error(
                    "--chunk_size must be a multiple of --shape_group_size "
                    "(otherwise shape groups straddle chunk files and are split "
                    "across DataLoader workers)"
                )
            if args.num_tasks % shape_group_size != 0:
                parser.error(
                    "--num_tasks must be a multiple of --shape_group_size "
                    "(otherwise the last shape group is incomplete)"
                )
        prior_config_file = args.config
        cfg = load_config(args.config)
        out_dir = Path(args.out_dir)
        split = args.split
        num_tasks, chunk_size, base_seed = (
            args.num_tasks,
            args.chunk_size,
            args.base_seed,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(
        cfg,
        out_dir,
        num_tasks,
        chunk_size,
        base_seed,
        split=split,
        shape_group_size=shape_group_size,
        dataset_name=dataset_name,
        prior_config_file=prior_config_file,
    )

    n = num_chunks(num_tasks, chunk_size)
    if args.job_index is not None:
        indices = range(args.job_index, n, args.num_jobs)
    elif args.chunk_index is not None:
        indices = [args.chunk_index]
    else:
        indices = range(n)
    for k in indices:
        start = k * chunk_size
        expected = min(start + chunk_size, num_tasks) - start
        existing = out_dir / f"chunk_{k:05d}.pt"
        if args.skip_existing_files and chunk_file_is_complete(existing, expected):
            print(f"Skipped {existing} (exists)")
            continue
        path = write_chunk(
            cfg,
            out_dir,
            k,
            num_tasks,
            chunk_size,
            base_seed,
            split=split,
            shape_group_size=shape_group_size,
            show_progress=True,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
