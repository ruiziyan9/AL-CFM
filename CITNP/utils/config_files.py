"""
Reading config files, with inheritance.

A YAML config file may name the file(s) it builds on with the top-level key
`extends` and then state only what differs:

    # CITNP/experiments/configs/oracle_d5_iw_a15.yaml
    extends: oracle_d3_iw_a15.yaml
    num_variables: [5]
    graph_degrees: {5: [2, 4, 6, 8]}

Rules:

- `extends` is one path or a list of paths, relative to the directory of the
  file that names them. It must be written on a single line,
  `extends: a.yaml` or `extends: [a.yaml, b.yaml]`, because the cluster script
  CITNP/experiments/bmrc/train.sh reads that line without a YAML parser.
- Only YAML files may use `extends`. A JSON config is still read, and may be
  named as a parent, but `extends` inside one is an error: train.sh could not
  see it, so its fingerprint would miss the parents.
- Parents are resolved recursively. The files are merged in the order parents
  (as listed), then the file itself; a later file wins.
- Top-level keys are replaced whole: a dictionary in the child fully replaces
  the parent's dictionary of the same key, it is not merged into it.
- The `extends` key itself is not part of the resolved mapping.
- A file without `extends` resolves to exactly its own contents.

`config_file_chain` lists the files a config is built from,
`load_config_mapping` returns the resolved mapping, and `config_fingerprint`
is a short code that changes when any of those files changes.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXTENDS_KEY = "extends"
YAML_SUFFIXES = (".yaml", ".yml")


def _read_mapping(path: Path) -> dict[str, Any]:
    """The contents of one config file, without following `extends`. An empty
    file is an empty mapping; anything else that is not a mapping raises
    ValueError naming the file."""
    with open(path) as f:
        if path.suffix in YAML_SUFFIXES:
            import yaml  # type: ignore[import-untyped]

            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a mapping of keys to values, "
            f"got {type(raw).__name__}"
        )
    return raw


def _parent_names_on_the_extends_line(path: Path) -> list[str]:
    """The parent paths as CITNP/experiments/bmrc/train.sh reads them: from
    every line starting with `extends:`, drop the comment, brackets, commas
    and quotes, and split at whitespace."""
    names: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(EXTENDS_KEY + ":"):
            value = line[len(EXTENDS_KEY) + 1 :].split("#", 1)[0]
            names.extend(re.sub(r"[\[\],'\"]", " ", value).split())
    return names


def _parent_paths(path: Path, mapping: dict[str, Any]) -> list[Path]:
    """Paths of the files named by `extends` in `mapping` (read from `path`),
    in the listed order; empty without `extends`."""
    if EXTENDS_KEY not in mapping:
        return []
    value = mapping[EXTENDS_KEY]
    names = [value] if isinstance(value, str) else value
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(
            f"{path}: {EXTENDS_KEY} must be one path or a list of paths, "
            f"got {value!r}"
        )
    if path.suffix not in YAML_SUFFIXES:
        raise ValueError(
            f"{path}: {EXTENDS_KEY} is only supported in YAML config files "
            "(CITNP/experiments/bmrc/train.sh cannot read it from a JSON "
            "file, so the config fingerprint would miss the parents)"
        )
    if _parent_names_on_the_extends_line(path) != names:
        raise ValueError(
            f"{path}: {EXTENDS_KEY} must be written on a single line, as "
            f"`{EXTENDS_KEY}: a.yaml` or `{EXTENDS_KEY}: [a.yaml, b.yaml]`, "
            "with paths free of spaces, commas, brackets, quotes and '#' "
            "(CITNP/experiments/bmrc/train.sh reads that line directly)"
        )
    return [path.parent / name for name in names]


def _chain_and_mappings(
    path: Path, files_being_resolved: tuple[Path, ...]
) -> list[tuple[Path, dict[str, Any]]]:
    """(file, its own contents) for every file `path` is built from, in merge
    order. `files_being_resolved` are the files whose `extends` led here."""
    identity = path.resolve()
    if identity in files_being_resolved:
        raise ValueError(
            f"{path}: {EXTENDS_KEY} cycle, this file is reached again from "
            "itself through the files it extends"
        )
    mapping = _read_mapping(path)
    chain: list[tuple[Path, dict[str, Any]]] = []
    for parent in _parent_paths(path, mapping):
        chain.extend(
            _chain_and_mappings(parent, (*files_being_resolved, identity))
        )
    chain.append((path, mapping))
    return chain


def config_file_chain(path: str | Path) -> list[Path]:
    """Every file the config at `path` is built from, in merge order: parents
    first (recursively, in the order they are listed), `path` itself last.

    A file reached through two different parents appears once per route, so
    merging the files in the returned order always gives the resolved mapping.
    Raises ValueError for an `extends` cycle.
    """
    return [file for file, _ in _chain_and_mappings(Path(path), ())]


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    """The resolved contents of the config file at `path` (`.yaml`, `.yml` or
    `.json`): its parents named by `extends` merged in order, then the file
    itself, top-level keys replaced whole, without the `extends` key.

    Raises ValueError, naming the file, if a file is not a mapping, if
    `extends` is malformed or used in a JSON file, or if the files extend
    each other in a cycle.
    """
    resolved: dict[str, Any] = {}
    for _, mapping in _chain_and_mappings(Path(path), ()):
        resolved.update(mapping)
    resolved.pop(EXTENDS_KEY, None)
    return resolved


def config_fingerprint(path: str | Path) -> str:
    """A short code (10 hexadecimal characters) computed from the contents of
    every file the config at `path` is built from (`config_file_chain`, joined
    in that order); it changes if any byte of any of them changes.

    This is the fingerprint CITNP/experiments/bmrc/train.sh computes. For a
    file without `extends` it equals `file_fingerprint` of
    CITNP/datasets/dataset_config.py, the sha1 of that file alone.
    """
    contents = b"".join(file.read_bytes() for file in config_file_chain(path))
    return hashlib.sha1(contents).hexdigest()[:10]
