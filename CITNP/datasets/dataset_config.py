"""
Configuration of pre-simulated datasets.

A dataset config (CITNP/experiments/configs/datasets/<dataset_name>.yaml) names
a prior config and the sizes of the train / validation / test splits that are
written once with create_mixed_regime_chunks.py and reused by mixed_train.py:

    dataset_name: oracle_d3_iw_motif__v1          # directory under the data root
    prior_config: CITNP/experiments/configs/oracle_d3_iw_motif.yaml  # repo-relative
    tasks_per_chunk_file: 4096
    tasks_per_shape_group: 256     # must be a multiple of the training batch size
    splits:
      train:      {num_tasks: 2048000, base_seed: 0}
      validation: {num_tasks: 12800,   base_seed: 0}
      test:       {num_tasks: 12800,   base_seed: 0}

The split called "validation" here is called "val" everywhere else
(SPLIT_IDS in mixed_regime_generator.py, the writer's --split, config.json);
`generator_split_name` converts. Simulated data live outside the git
checkout, under a data root given by --data_root or the environment variable
ALCFM_DATA_ROOT: the files of a split are in <data_root>/<dataset_name>/<split>.
"""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENVIRONMENT_VARIABLE = "ALCFM_DATA_ROOT"
# Split names in a dataset config -> split names of the generator.
GENERATOR_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
DATASET_CONFIG_KEYS = {
    "dataset_name",
    "prior_config",
    "tasks_per_chunk_file",
    "tasks_per_shape_group",
    "splits",
}


@dataclass(frozen=True)
class SplitSettings:
    """Size and seed of one split."""

    num_tasks: int
    base_seed: int


@dataclass(frozen=True)
class PresimulatedDatasetConfig:
    """A validated dataset config.

    Args:
    ----------
    dataset_name : str
        Directory of the dataset under the data root.

    prior_config : str
        Path of the prior (MixedRegimeConfig) YAML, relative to the repository
        root; see `prior_config_path`.

    tasks_per_chunk_file : int
        Tasks per chunk file; a multiple of tasks_per_shape_group.

    tasks_per_shape_group : int
        Consecutive tasks sharing one batch shape (D, N_ctx, G); must be a
        multiple of the training batch size for padding-free batches.

    splits : dict[str, SplitSettings]
        Keyed by "train", "validation" and "test" (any subset); num_tasks is a
        multiple of tasks_per_shape_group.
    """

    dataset_name: str
    prior_config: str
    tasks_per_chunk_file: int
    tasks_per_shape_group: int
    splits: dict[str, SplitSettings]


def file_fingerprint(path: str | Path) -> str:
    """A short code (10 hexadecimal characters) computed from the file's
    contents; it changes if any byte of the file changes."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


def generator_split_name(split: str) -> str:
    """Split name used by the generator ("validation" -> "val")."""
    if split not in GENERATOR_SPLIT_NAMES:
        raise ValueError(
            f"Unknown split {split!r}; choose from {list(GENERATOR_SPLIT_NAMES)}"
        )
    return GENERATOR_SPLIT_NAMES[split]


def prior_config_path(dataset_config: PresimulatedDatasetConfig) -> Path:
    """Absolute path of the dataset's prior config."""
    return REPOSITORY_ROOT / dataset_config.prior_config


def load_presimulated_dataset_config(
    path: str | Path,
) -> PresimulatedDatasetConfig:
    """Read and validate a dataset config; raises ValueError (or
    FileNotFoundError for a missing file) with a message naming the problem."""
    # Imported here: the writer module imports this one.
    from CITNP.datasets.create_mixed_regime_chunks import load_config

    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = set(raw) - DATASET_CONFIG_KEYS
    missing = DATASET_CONFIG_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )

    name = raw["dataset_name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(
            f"{path}: dataset_name must be a plain directory name, got {name!r}"
        )
    group = raw["tasks_per_shape_group"]
    chunk = raw["tasks_per_chunk_file"]
    for key, value in (
        ("tasks_per_shape_group", group),
        ("tasks_per_chunk_file", chunk),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    if chunk % group != 0:
        raise ValueError(
            f"{path}: tasks_per_chunk_file={chunk} must be a multiple of "
            f"tasks_per_shape_group={group}"
        )

    if not isinstance(raw["splits"], dict) or not raw["splits"]:
        raise ValueError(f"{path}: splits must be a non-empty mapping")
    splits = {}
    for split, settings in raw["splits"].items():
        if split not in GENERATOR_SPLIT_NAMES:
            raise ValueError(
                f"{path}: unknown split {split!r}; "
                f"choose from {list(GENERATOR_SPLIT_NAMES)}"
            )
        if not isinstance(settings, dict) or set(settings) != {
            "num_tasks",
            "base_seed",
        }:
            raise ValueError(
                f"{path}: split {split!r} needs exactly num_tasks and base_seed"
            )
        num_tasks, base_seed = settings["num_tasks"], settings["base_seed"]
        if not isinstance(num_tasks, int) or num_tasks <= 0 or num_tasks % group:
            raise ValueError(
                f"{path}: split {split!r}: num_tasks={num_tasks!r} must be a "
                f"positive multiple of tasks_per_shape_group={group}"
            )
        if not isinstance(base_seed, int) or base_seed < 0:
            raise ValueError(f"{path}: split {split!r}: base_seed must be >= 0")
        splits[split] = SplitSettings(num_tasks, base_seed)

    config = PresimulatedDatasetConfig(
        dataset_name=name,
        prior_config=str(raw["prior_config"]),
        tasks_per_chunk_file=chunk,
        tasks_per_shape_group=group,
        splits=splits,
    )
    prior_path = prior_config_path(config)
    if not prior_path.is_file():
        raise FileNotFoundError(
            f"{path}: prior_config {config.prior_config} not found "
            f"(looked for {prior_path})"
        )
    if load_config(prior_path).qry_mode != "grouped":
        raise ValueError(
            f"{prior_path}: pre-simulated datasets need qry_mode: grouped"
        )
    return config


def _repository_working_tree() -> Path:
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return Path(top.decode().strip()).resolve()
    except Exception:
        return REPOSITORY_ROOT


def resolve_data_root(data_root: str | None) -> Path:
    """The directory holding simulated datasets: `data_root`, else the
    environment variable ALCFM_DATA_ROOT, else an error. Refuses a path inside
    this repository's git working tree (simulated data must not be committed)."""
    value = data_root or os.environ.get(DATA_ROOT_ENVIRONMENT_VARIABLE)
    if not value:
        raise ValueError(
            "No data root: pass --data_root or set the environment variable "
            f"{DATA_ROOT_ENVIRONMENT_VARIABLE}"
        )
    root = Path(value).expanduser().resolve()
    tree = _repository_working_tree()
    if root == tree or tree in root.parents:
        raise ValueError(
            f"Data root {root} is inside the git working tree {tree}; keep "
            "simulated data outside the repository"
        )
    return root


def split_directory(
    dataset_config: PresimulatedDatasetConfig,
    data_root: str | Path,
    split: str,
) -> Path:
    """<data_root>/<dataset_name>/<split>, with the split named as in the
    dataset config ("train", "validation" or "test")."""
    generator_split_name(split)
    return Path(data_root) / dataset_config.dataset_name / split
