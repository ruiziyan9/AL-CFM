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
# modules (plot_mixed_prior, tests) import load_config from here and must keep
# their CUDA setting. The mechanisms in functions_generator.py choose their
# device when they are constructed, so this is early enough.
if __name__ == "__main__" and "--allow_cuda" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import dataclasses
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from CITNP.datasets.mixed_regime_generator import (
    SOURCE_CHUNK,
    SPLIT_IDS,
    MixedRegimeConfig,
    generate_task,
    make_key,
    num_chunks,
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
) -> None:
    """Write config.json: the config, split and seed, git commit and runtime."""
    info = {
        "config": dataclasses.asdict(cfg),
        "num_tasks": num_tasks,
        "chunk_size": chunk_size,
        "base_seed": base_seed,
        "split": split,
        "split_id": SPLIT_IDS[split],
        "shape_group_size": shape_group_size,
        "git_commit": _git_commit(),
        # Tasks are reproducible only on the same device type and versions.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }

    def _write(p):
        with open(p, "w") as f:
            json.dump(info, f, indent=2)

    _atomic_write(out_dir / "config.json", _write)


def group_shape(
    cfg: MixedRegimeConfig, split_id: int, base_seed: int, group_index: int
) -> tuple[int, int, int | None]:
    """(D, n_ctx, n_groups) shared by all tasks of one shape group.

    Drawn from its own key (i2 = 1 keeps it apart from task keys, whose i2 is
    0): D from cfg.num_variables, then n_ctx ~ U{n_ctx_range}, then (grouped
    qry_mode only) the number of question groups G from cfg.qry_groups.
    n_groups is None in per-row mode, which consumes exactly the old draws.
    """
    rng = np.random.default_rng(
        list(make_key(SOURCE_CHUNK, split_id, base_seed, group_index, 1, 0))
    )
    D = int(rng.choice(cfg.num_variables))
    lo, hi = cfg.n_ctx_range
    n_ctx = int(rng.integers(lo, hi + 1))
    n_groups = None
    if cfg.qry_mode == "grouped":
        n_groups = int(rng.choice(cfg.qry_groups))
    return D, n_ctx, n_groups


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
            D, n_ctx, n_groups = group_shape(
                cfg, split_id, base_seed, idx // shape_group_size
            )
            task = generate_task(
                cfg, key, num_variables=D, n_ctx=n_ctx, n_groups=n_groups
            )
        else:
            task = generate_task(cfg, key)
        tasks.append(dataclasses.asdict(task))
    path = out_dir / f"chunk_{chunk_index:05d}.pt"
    _atomic_write(path, lambda p: torch.save(tasks, p))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", required=True, help="yaml or json config")
    parser.add_argument("--num_tasks", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=list(SPLIT_IDS),
        help="Data split; train / val / test never share tasks",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=None,
        help="Write only this chunk (e.g. $SLURM_ARRAY_TASK_ID); default: all",
    )
    parser.add_argument(
        "--shape_group_size",
        type=int,
        default=256,
        help="Consecutive tasks sharing one (D, N_ctx) for padding-free "
        "batches; use a multiple of the batch size. 0 disables grouping.",
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
    if args.base_seed < 0:
        parser.error("--base_seed must be >= 0")
    if args.shape_group_size < 0:
        parser.error("--shape_group_size must be >= 0")
    if args.shape_group_size > 0:
        if args.chunk_size % args.shape_group_size != 0:
            parser.error(
                "--chunk_size must be a multiple of --shape_group_size "
                "(otherwise shape groups straddle chunk files and are split "
                "across DataLoader workers)"
            )
        if args.num_tasks % args.shape_group_size != 0:
            parser.error(
                "--num_tasks must be a multiple of --shape_group_size "
                "(otherwise the last shape group is incomplete)"
            )

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(
        cfg,
        out_dir,
        args.num_tasks,
        args.chunk_size,
        args.base_seed,
        split=args.split,
        shape_group_size=args.shape_group_size,
    )

    n = num_chunks(args.num_tasks, args.chunk_size)
    indices = range(n) if args.chunk_index is None else [args.chunk_index]
    for k in indices:
        path = write_chunk(
            cfg,
            out_dir,
            k,
            args.num_tasks,
            args.chunk_size,
            args.base_seed,
            split=args.split,
            shape_group_size=args.shape_group_size,
            show_progress=True,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
