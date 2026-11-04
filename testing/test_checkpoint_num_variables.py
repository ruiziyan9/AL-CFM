"""Tests for reading a checkpoint's number of variables and training prior
(model_api.training_prior_config_path, checkpoint_num_variables, load_model)
and for the copy mixed_train writes (record_training_prior_config). CPU, no
training."""

import argparse
import sys
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from CITNP.evaluation.model_api import (
    FALLBACK_NUM_NODES,
    checkpoint_num_variables,
    load_model,
    training_prior_config_path,
)
from CITNP.experiments.mixed_train import record_training_prior_config
from CITNP.models.mixed_context_model import resolve_model_class

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_intervention_probes import PRIOR, tiny_model_yaml  # noqa: E402
from test_mixed_context_model import tiny_config  # noqa: E402


def save_model(directory: Path, encoding: str, num_nodes: int) -> Path:
    """Random tiny model of `encoding` built for num_nodes variables, saved
    as directory/best_model.pt with its model YAML directory/model.yaml."""
    directory.mkdir(parents=True, exist_ok=True)
    model_class, _ = resolve_model_class(encoding)
    cfg = tiny_config()
    cfg.num_nodes = num_nodes
    torch.manual_seed(0)
    ckpt = directory / "best_model.pt"
    torch.save({"model_state_dict": model_class(cfg).state_dict()}, ckpt)
    tiny_model_yaml(directory / "model.yaml", encoding)
    return ckpt


def write_training_prior(directory: Path, num_variables: list[int]) -> Path:
    prior = dict(PRIOR, num_variables=num_variables)
    prior["graph_degrees"] = {D: [1, 2] for D in num_variables}
    path = directory / "training_prior_config.yaml"
    path.write_text(yaml.safe_dump(prior))
    return path


def test_nothing_recorded_keeps_the_old_default(tmp_path):
    ckpt = save_model(tmp_path, "r1", 3)
    assert training_prior_config_path(ckpt) is None
    assert checkpoint_num_variables(ckpt) == (None, None, "nothing recorded")
    model = load_model(str(ckpt), str(tmp_path / "model.yaml"), "cpu")
    assert model.num_nodes == FALLBACK_NUM_NODES


def test_extra_node_index_embedding_gives_the_number_of_variables(tmp_path):
    ckpt = save_model(tmp_path, "extra_node", 4)
    model_config = str(tmp_path / "model.yaml")
    state = torch.load(ckpt)["model_state_dict"]
    built, training, _ = checkpoint_num_variables(ckpt, state)
    assert (built, training) == (4, None)
    assert load_model(str(ckpt), model_config, "cpu").encoder.num_nodes == 4
    # fewer variables than the model was built for: allowed, same weights
    fewer = load_model(str(ckpt), model_config, "cpu", num_nodes=3)
    assert fewer.encoder.num_nodes == 4
    with pytest.raises(ValueError, match="disagrees"):
        load_model(str(ckpt), model_config, "cpu", num_nodes=5)


def test_training_prior_config_gives_the_number_of_variables(tmp_path):
    ckpt = save_model(tmp_path, "layerwise", 3)
    prior = write_training_prior(tmp_path, [3, 4])
    assert training_prior_config_path(ckpt) == prior.resolve()
    built, training, source = checkpoint_num_variables(ckpt)
    assert (built, training) == (4, [3, 4]) and "training prior" in source
    model_config = str(tmp_path / "model.yaml")
    assert load_model(str(ckpt), model_config, "cpu").num_nodes == 4
    assert load_model(str(ckpt), model_config, "cpu", num_nodes=3).num_nodes == 4
    with pytest.raises(ValueError, match="trained on num_variables"):
        load_model(str(ckpt), model_config, "cpu", num_nodes=2)


def test_prior_and_index_embedding_must_agree(tmp_path):
    ckpt = save_model(tmp_path, "extra_node", 3)
    write_training_prior(tmp_path, [3, 4])
    state = torch.load(ckpt)["model_state_dict"]
    with pytest.raises(ValueError, match="index embedding"):
        checkpoint_num_variables(ckpt, state)


def test_mixed_train_records_the_prior_verbatim(tmp_path):
    source = tmp_path / "prior.yaml"
    source.write_text(yaml.safe_dump(PRIOR))
    run = tmp_path / "run"
    run.mkdir()
    record_training_prior_config(argparse.Namespace(data_config=str(source)), run)
    copy = run / "training_prior_config.yaml"
    assert copy.read_bytes() == source.read_bytes()
    assert training_prior_config_path(run / "best_model.pt") == copy.resolve()


def test_diagnostic_cli_defaults_to_the_training_prior(tmp_path):
    import json

    import pandas as pd

    from CITNP.evaluation.__main__ import main

    run = tmp_path / "run"
    ckpt = save_model(run, "extra_node", 3)
    (run / "training_prior_config.yaml").write_text(yaml.safe_dump(PRIOR))
    out = tmp_path / "probes.csv"
    main(["probe-intervention", "--ckpt", str(ckpt), "--model_config",
          str(run / "model.yaml"), "--n_tasks", "4", "--rows_per_task", "8",
          "--n_folds", "2", "--device", "cpu", "--out", str(out)])  # fmt: skip
    assert len(pd.read_csv(out)) > 0
    meta = json.loads((tmp_path / "probes_meta.json").read_text())
    assert meta["args"]["data_config"].endswith("training_prior_config.yaml")
    assert meta["num_variables"] == 3


def test_diagnostic_cli_needs_a_prior_when_none_is_recorded(tmp_path):
    from CITNP.evaluation.__main__ import main

    ckpt = save_model(tmp_path, "r1", 3)
    with pytest.raises(SystemExit):
        main(["probe-intervention", "--ckpt", str(ckpt), "--model_config",
              str(tmp_path / "model.yaml"), "--device", "cpu",
              "--out", str(tmp_path / "p.csv")])  # fmt: skip


def test_diagnostic_cli_rejects_a_disagreeing_num_variables(tmp_path):
    from CITNP.evaluation.__main__ import main

    run = tmp_path / "run"
    ckpt = save_model(run, "extra_node", 3)
    (run / "training_prior_config.yaml").write_text(yaml.safe_dump(PRIOR))
    with pytest.raises(SystemExit):
        main(["probe-intervention", "--ckpt", str(ckpt), "--model_config",
              str(run / "model.yaml"), "--num_variables", "4",
              "--device", "cpu", "--out", str(tmp_path / "p.csv")])  # fmt: skip
