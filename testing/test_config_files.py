"""Config inheritance with `extends` (CITNP/utils/config_files.py) and the
config fingerprint of CITNP/experiments/bmrc/train.sh, which must cover every
file a config is built from."""

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from CITNP.datasets.create_mixed_regime_chunks import load_config
from CITNP.datasets.mixed_regime_generator import MixedRegimeConfig
from CITNP.experiments.mixed_train import MODEL_KEYS
from CITNP.utils.config_files import (
    config_file_chain,
    config_fingerprint,
    load_config_mapping,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "CITNP" / "experiments" / "configs"
TRAIN_SCRIPT = REPO / "CITNP" / "experiments" / "bmrc" / "train.sh"
SHIPPED_CONFIGS = sorted(CONFIGS.glob("*.yaml"))
CHILD = CONFIGS / "oracle_d5_iw_a15.yaml"
PARENT = CONFIGS / "oracle_d3_iw_a15.yaml"
MODEL_CONFIG = CONFIGS / "smoke_d3_model.yaml"

needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha1sum") is None,
    reason="bash and sha1sum are needed to run train.sh",
)


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _is_model_config(path: Path) -> bool:
    return set(load_config_mapping(path)) <= set(MODEL_KEYS)


def _train_script_dry_run(
    tmp_path: Path,
    data_config: Path,
    model_config: Path,
    extra_environment: dict[str, str] | None = None,
):
    """Run train.sh with DRY_RUN=1 on the two configs; returns the completed
    process (not checked). REPO does not exist, so the script stays in the
    current directory, and OUT is empty, so the run is a fresh one.
    `extra_environment` adds further train.sh variables."""
    environment = {
        "PATH": os.environ["PATH"],
        "USER": "tester",
        "DRY_RUN": "1",
        "REPO": str(tmp_path / "no_such_checkout"),
        "OUT": str(tmp_path / "out"),
        "DATA_CFG": str(data_config),
        "MODEL_CFG": str(model_config),
        **(extra_environment or {}),
    }
    return subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
    )


def _train_script_data_fingerprint(tmp_path: Path, data_config: Path) -> str:
    """The data-config fingerprint train.sh computes for `data_config`."""
    result = _train_script_dry_run(tmp_path, data_config, MODEL_CONFIG)
    assert result.returncode == 0, result.stderr
    match = re.search(r"^DATA_CFG=.* \(fingerprint (\w+)\)$", result.stdout, re.M)
    assert match, result.stdout
    assert "sha1" not in result.stdout  # the label is "fingerprint"
    return match.group(1)


def _train_script_data_chain(tmp_path: Path, data_config: Path) -> list[Path]:
    """The files train.sh reports the data config is built from."""
    result = _train_script_dry_run(tmp_path, data_config, MODEL_CONFIG)
    assert result.returncode == 0, result.stderr
    data_section = result.stdout.split("DATA_CFG=", 1)[1].split("MODEL_CFG=")[0]
    listed = re.findall(r"^#     (.+)$", data_section, re.M)
    return [Path(line) for line in listed]


# ------------------------------------------------------------- merge rules


def test_child_key_overrides_parent_and_other_parent_keys_survive(tmp_path):
    _write(tmp_path, "parent.yaml", "alpha: 1\nbeta: [1, 2]\ngamma: kept\n")
    child = _write(tmp_path, "child.yaml", "extends: parent.yaml\nbeta: [3]\n")
    assert load_config_mapping(child) == {
        "alpha": 1,
        "beta": [3],
        "gamma": "kept",
    }
    assert config_file_chain(child) == [tmp_path / "parent.yaml", child]


def test_later_parent_wins_and_child_wins_over_both(tmp_path):
    _write(tmp_path, "first.yaml", "alpha: first\nbeta: first\ngamma: first\n")
    _write(tmp_path, "second.yaml", "beta: second\ngamma: second\n")
    child = _write(
        tmp_path,
        "child.yaml",
        "extends: [first.yaml, second.yaml]  # two parents\ngamma: child\n",
    )
    assert load_config_mapping(child) == {
        "alpha": "first",
        "beta": "second",
        "gamma": "child",
    }
    assert config_file_chain(child) == [
        tmp_path / "first.yaml",
        tmp_path / "second.yaml",
        child,
    ]


def test_dictionary_value_is_replaced_whole_not_merged(tmp_path):
    _write(tmp_path, "parent.yaml", "graph_degrees: {3: [1, 2, 3], 4: [2]}\n")
    child = _write(
        tmp_path,
        "child.yaml",
        "extends: parent.yaml\ngraph_degrees: {5: [2, 4]}\n",
    )
    assert load_config_mapping(child) == {"graph_degrees": {5: [2, 4]}}


def test_parents_are_resolved_recursively_relative_to_the_naming_file(tmp_path):
    _write(tmp_path, "shared/grandparent.yaml", "alpha: 1\nbeta: 1\ngamma: 1\n")
    _write(tmp_path, "shared/parent.yaml", "extends: grandparent.yaml\nbeta: 2\n")
    child = _write(
        tmp_path, "child.yaml", "extends: shared/parent.yaml\ngamma: 3\n"
    )
    assert load_config_mapping(child) == {"alpha": 1, "beta": 2, "gamma": 3}
    assert "extends" not in load_config_mapping(child)
    assert [file.name for file in config_file_chain(child)] == [
        "grandparent.yaml",
        "parent.yaml",
        "child.yaml",
    ]


def test_extends_in_a_json_config_is_rejected(tmp_path):
    # train.sh cannot read `extends` from JSON, so its fingerprint would
    # silently miss the parents.
    _write(tmp_path, "parent.yaml", "alpha: 1\n")
    child = _write(
        tmp_path, "child.json", '{"extends": "parent.yaml", "beta": 2}'
    )
    with pytest.raises(ValueError, match=r"child\.json.*only supported in YAML"):
        load_config_mapping(child)
    with pytest.raises(ValueError, match=r"child\.json"):
        config_file_chain(child)
    # A JSON config without extends still loads, also as a YAML file's parent.
    plain = _write(tmp_path, "plain.json", '{"alpha": 1, "beta": 1}')
    assert load_config_mapping(plain) == {"alpha": 1, "beta": 1}
    yaml_child = _write(
        tmp_path, "yaml_child.yaml", "extends: plain.json\nbeta: 2\n"
    )
    assert load_config_mapping(yaml_child) == {"alpha": 1, "beta": 2}


def test_cycle_raises_value_error_naming_the_file(tmp_path):
    first = _write(tmp_path, "first.yaml", "extends: nested/second.yaml\n")
    _write(tmp_path, "nested/second.yaml", "extends: ../first.yaml\n")
    with pytest.raises(ValueError, match=r"first\.yaml.*cycle"):
        load_config_mapping(first)
    with pytest.raises(ValueError, match="cycle"):
        config_file_chain(first)
    own_parent = _write(tmp_path, "own_parent.yaml", "extends: own_parent.yaml\n")
    with pytest.raises(ValueError, match=r"own_parent\.yaml"):
        load_config_mapping(own_parent)


def test_file_that_is_not_a_mapping_raises_value_error_naming_the_file(tmp_path):
    path = _write(tmp_path, "a_list.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError, match=r"a_list\.yaml"):
        load_config_mapping(path)


def test_extends_written_over_several_lines_is_rejected(tmp_path):
    # train.sh reads the `extends:` line as text, so a form it cannot read
    # must fail in Python instead of giving a fingerprint that misses parents.
    _write(tmp_path, "parent.yaml", "alpha: 1\n")
    child = _write(tmp_path, "child.yaml", "extends:\n  - parent.yaml\n")
    with pytest.raises(ValueError, match="single line"):
        load_config_mapping(child)
    not_a_path = _write(tmp_path, "not_a_path.yaml", "extends: 3\n")
    with pytest.raises(ValueError, match=r"not_a_path\.yaml"):
        load_config_mapping(not_a_path)


# ----------------------------------------------------- the shipped configs


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_file_without_extends_resolves_to_plain_yaml(path):
    raw = yaml.safe_load(path.read_text())
    if "extends" in raw:
        pytest.skip("this config uses extends")
    resolved = load_config_mapping(path)
    assert resolved == raw
    assert list(resolved) == list(raw)  # same key order
    assert config_file_chain(path) == [path]


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_shipped_config_loads_and_data_configs_are_unchanged(path):
    resolved = load_config_mapping(path)
    assert isinstance(resolved, dict) and resolved
    raw = yaml.safe_load(path.read_text())
    if _is_model_config(path) or "extends" in raw:
        return
    # A data config without extends: the same prior as before inheritance
    # existed, when the file's own contents were passed to MixedRegimeConfig.
    assert load_config(path) == MixedRegimeConfig(**raw)


def test_five_variable_config_differs_from_its_parent_in_stated_fields_only():
    child = dataclasses.asdict(load_config(CHILD))  # passes validation
    parent = dataclasses.asdict(load_config(PARENT))
    assert yaml.safe_load(CHILD.read_text()) == {
        "extends": "oracle_d3_iw_a15.yaml",
        "num_variables": [5],
        "graph_type": ["ER", "ER", "SF", "FC"],
        "graph_degrees": {5: [3, 5, 8, 9]},
        "k_max": 5,
        "p_k0": 0.25,
        "p_geom": 0.3,
        "n_ctx_range": [64, 360],
        "n_qry": 60,
        "qry_mode": "grouped",
        "qry_groups": [1, 2, 3, 4, 5],
        "n_obs_min": 32,
    }
    # qry_mode and n_obs_min restate the parent's values.
    differing = {key for key in child if child[key] != parent[key]}
    assert differing == {
        "num_variables",
        "graph_type",
        "graph_degrees",
        "k_max",
        "p_k0",
        "p_geom",
        "n_ctx_range",
        "n_qry",
        "qry_groups",
    }
    assert list(child["num_variables"]) == [5]
    assert {int(d): list(v) for d, v in child["graph_degrees"].items()} == {
        5: [3, 5, 8, 9]
    }
    assert config_file_chain(CHILD) == [PARENT, CHILD]


# ------------------------------------------------- train.sh agrees with Python


@needs_bash
def test_train_script_chain_matches_python_chain(tmp_path):
    assert _train_script_data_chain(tmp_path, CHILD) == config_file_chain(CHILD)

    # Two listed parents, one of them with a parent of its own in another
    # directory, a comment on the extends line and quoted paths.
    _write(tmp_path, "base/grandparent.yaml", "alpha: 1\n")
    _write(tmp_path, "base/first.yaml", "extends: grandparent.yaml\nbeta: 1\n")
    _write(tmp_path, "second.yaml", "beta: 2\n")
    child = _write(
        tmp_path,
        "child.yaml",
        "# header\nextends: ['base/first.yaml', \"second.yaml\"]  # parents\n"
        "gamma: 3\n",
    )
    python_chain = config_file_chain(child)
    assert [file.name for file in python_chain] == [
        "grandparent.yaml",
        "first.yaml",
        "second.yaml",
        "child.yaml",
    ]
    script_chain = _train_script_data_chain(tmp_path, child)
    assert [file.resolve() for file in script_chain] == [
        file.resolve() for file in python_chain
    ]


@needs_bash
@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_fingerprint_of_file_without_extends_is_unchanged(tmp_path, path):
    if "extends" in yaml.safe_load(path.read_text()):
        pytest.skip("this config uses extends")
    old_fingerprint = subprocess.run(
        f"sha1sum '{path}' | cut -c1-10",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert old_fingerprint == hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    assert _train_script_data_fingerprint(tmp_path, path) == old_fingerprint
    assert config_fingerprint(path) == old_fingerprint
    # No chain is listed for a file that stands alone.
    assert _train_script_data_chain(tmp_path, path) == []


@needs_bash
def test_fingerprint_covers_the_whole_chain(tmp_path):
    parent = _write(tmp_path, "parent.yaml", "alpha: 1\nbeta: 1\n")
    child = _write(tmp_path, "child.yaml", "extends: parent.yaml\nbeta: 2\n")
    before = _train_script_data_fingerprint(tmp_path, child)
    joined = b"".join(file.read_bytes() for file in config_file_chain(child))
    assert before == hashlib.sha1(joined).hexdigest()[:10]
    assert before == config_fingerprint(child)  # Python computes the same

    parent.write_text("alpha: 5\nbeta: 1\n")  # edit the parent only
    after = _train_script_data_fingerprint(tmp_path, child)
    assert after != before
    assert _train_script_data_fingerprint(tmp_path, parent) != before


@needs_bash
def test_train_script_aborts_on_cycle_and_on_missing_parent(tmp_path):
    first = _write(tmp_path, "first.yaml", "extends: nested/second.yaml\n")
    _write(tmp_path, "nested/second.yaml", "extends: ../first.yaml\n")
    result = _train_script_dry_run(tmp_path, first, MODEL_CONFIG)
    assert result.returncode != 0 and "cycle" in result.stderr

    orphan = _write(tmp_path, "orphan.yaml", "extends: no_such_parent.yaml\n")
    result = _train_script_dry_run(tmp_path, orphan, MODEL_CONFIG)
    assert result.returncode != 0 and "no_such_parent.yaml" in result.stderr


@needs_bash
def test_skip_oracle_bank_trains_without_a_bank(tmp_path):
    result = _train_script_dry_run(
        tmp_path, CHILD, MODEL_CONFIG, {"SKIP_ORACLE_BANK": "1"}
    )
    assert result.returncode == 0, result.stderr
    assert "# oracle bank: skipped (SKIP_ORACLE_BANK=1)" in result.stdout
    assert not re.search(r"^BANK=", result.stdout, re.M)
    assert "--oracle_bank" not in result.stdout
    assert "--oracle_every" not in result.stdout
    assert "CITNP.evaluation.oracle_bank" not in result.stdout  # no build
    assert "CITNP.experiments.mixed_train" in result.stdout


@needs_bash
def test_oracle_bank_is_used_unless_skipped(tmp_path):
    unset = _train_script_dry_run(tmp_path, PARENT, MODEL_CONFIG)
    assert unset.returncode == 0, unset.stderr
    assert "skipped" not in unset.stdout
    bank = re.search(r"^BANK=(.+)$", unset.stdout, re.M)
    assert bank, unset.stdout
    assert f"--oracle_bank {bank.group(1)} \\\n" in unset.stdout
    assert "--oracle_every 2500 \\\n" in unset.stdout
    assert "CITNP.evaluation.oracle_bank" in unset.stdout  # bank build shown

    # Only the value 1 skips; 0 is the default behaviour, line for line.
    zero = _train_script_dry_run(
        tmp_path, PARENT, MODEL_CONFIG, {"SKIP_ORACLE_BANK": "0"}
    )
    assert zero.returncode == 0 and zero.stdout == unset.stdout


# ------------------------------------------------------ readers use extends


def test_model_config_with_extends_is_checked_on_the_resolved_keys(tmp_path):
    from CITNP.experiments.mixed_train import apply_model_config, parse_args

    child = _write(
        tmp_path,
        "wider_model.yaml",
        f"extends: {MODEL_CONFIG}\nd_model: 128\n",
    )
    base = yaml.safe_load(MODEL_CONFIG.read_text())
    data_config = ["--data_config", str(CONFIGS / "smoke_d3.yaml")]
    args = parse_args([*data_config, "--model_config", str(child), "--no_wandb"])
    apply_model_config(args)
    assert args.d_model == 128
    assert args.nhead == base["nhead"]

    _write(tmp_path, "bad_parent.yaml", "not_a_model_key: 1\n")
    bad_child = _write(
        tmp_path, "bad_child.yaml", "extends: bad_parent.yaml\nd_model: 64\n"
    )
    args = parse_args(
        [*data_config, "--model_config", str(bad_child), "--no_wandb"]
    )
    with pytest.raises(ValueError, match="not_a_model_key"):
        apply_model_config(args)


def test_run_directory_records_the_resolved_data_config(tmp_path):
    from CITNP.experiments.mixed_train import build_trainer, parse_args

    _write(
        tmp_path,
        "smaller_queries.yaml",
        f"extends: {CONFIGS / 'smoke_d3.yaml'}\nn_qry: 12\n",
    )
    data_config_path = tmp_path / "smaller_queries.yaml"
    args = parse_args(
        [
            "--data_config", str(data_config_path),
            "--model_config", str(MODEL_CONFIG),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "1", "--epochs", "1",
            "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--results_path", str(tmp_path), "--run_name", "r",
        ]
    )  # fmt: skip
    data_config = load_config(data_config_path)
    build_trainer(args, data_config)
    recorded = yaml.safe_load(
        (tmp_path / "results" / "r" / "resolved_data_config.yaml").read_text()
    )
    assert recorded["n_qry"] == 12
    # The record alone rebuilds the prior the run was trained on.
    assert MixedRegimeConfig(**recorded) == data_config


def test_data_directory_records_the_fingerprint_of_the_whole_chain(tmp_path):
    from CITNP.datasets.create_mixed_regime_chunks import write_config
    from CITNP.datasets.dataset_config import file_fingerprint

    parent = _write(
        tmp_path, "parent.yaml", (CONFIGS / "smoke_d3.yaml").read_text()
    )
    child = _write(tmp_path, "child.yaml", "extends: parent.yaml\nn_qry: 12\n")
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    def recorded_fingerprint() -> str:
        write_config(
            load_config(child),
            data_directory,
            num_tasks=8,
            chunk_size=4,
            base_seed=0,
            split="train",
            prior_config_file=child,
        )
        info = json.loads((data_directory / "config.json").read_text())
        return info["prior_config_fingerprint"]

    before = recorded_fingerprint()
    assert before == config_fingerprint(child)
    assert before != file_fingerprint(child)  # not the child file alone

    # Editing only a comment of the parent changes the fingerprint, but the
    # prior is the same, so the existing directory is still accepted.
    parent.write_text("# a new comment\n" + parent.read_text())
    after = recorded_fingerprint()
    assert after == config_fingerprint(child) != before

    # A parent edit that changes the prior is still refused, as before.
    changed_prior = parent.read_text().replace(
        "n_ctx_range: [64, 256]", "n_ctx_range: [64, 128]"
    )
    assert changed_prior != parent.read_text()
    parent.write_text(changed_prior)
    with pytest.raises(ValueError, match="different generation settings"):
        recorded_fingerprint()
