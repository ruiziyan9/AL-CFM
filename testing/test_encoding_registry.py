"""Tests for the intervention-encoding switch (resolve_model_class and the
mixed_train / model_api wiring)."""

import argparse
import sys
from pathlib import Path

import pytest
import torch

from CITNP.evaluation.model_api import load_model
from CITNP.experiments.mixed_train import MODEL_KEYS, apply_model_config
from CITNP.models.mixed_context_model import (
    DEFAULT_INTERVENTION_ENCODING,
    INTERVENTION_ENCODINGS,
    LayerwiseMixedContextMoGModel,
    MixedContextMoGModel,
    resolve_model_class,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import tiny_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "CITNP" / "experiments" / "configs"


def _args(**kw) -> argparse.Namespace:
    base = {k: None for k in MODEL_KEYS}
    base["model_config"] = None
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_is_r1():
    cls, name = resolve_model_class()
    assert DEFAULT_INTERVENTION_ENCODING == "r1"
    assert cls is MixedContextMoGModel and name == "r1"


@pytest.mark.parametrize("name", sorted(INTERVENTION_ENCODINGS))
def test_every_encoding_resolves(name):
    cls, got = resolve_model_class(intervention_encoding=name)
    assert cls is INTERVENTION_ENCODINGS[name] and got == name


@pytest.mark.parametrize(
    "cls_name, expected",
    [
        ("MixedContextMoGModel", "r1"),
        ("LayerwiseMixedContextMoGModel", "layerwise"),
        ("MixedContextModel", "MixedContextModel"),  # Gaussian head, unregistered
    ],
)
def test_legacy_model_class(cls_name, expected):
    cls, name = resolve_model_class(model_class=cls_name)
    assert cls.__name__ == cls_name and name == expected


def test_agreeing_keys_ok():
    cls, name = resolve_model_class("layerwise", "LayerwiseMixedContextMoGModel")
    assert cls is LayerwiseMixedContextMoGModel and name == "layerwise"


def test_contradicting_keys_raise():
    with pytest.raises(ValueError, match="contradicts"):
        resolve_model_class("r1", "LayerwiseMixedContextMoGModel")


@pytest.mark.parametrize(
    "kw",
    [
        {"intervention_encoding": "nope"},
        {"model_class": "Nope"},
        {"model_class": "torch"},  # a module in the namespace, not a model
        {"model_class": "build_mlp"},  # a function, not a model
    ],
)
def test_unknown_names_raise(kw):
    with pytest.raises(ValueError, match="Unknown"):
        resolve_model_class(**kw)


def test_yaml_sets_encoding(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("d_model: 16\nintervention_encoding: layerwise\n")
    a = _args(model_config=str(p))
    apply_model_config(a)
    assert a.d_model == 16 and a.intervention_encoding == "layerwise"


def test_yaml_cli_conflict_raises(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("model_class: LayerwiseMixedContextMoGModel\n")
    a = _args(model_config=str(p), model_class="MixedContextMoGModel")
    with pytest.raises(ValueError, match="contradicts"):
        apply_model_config(a)


def test_yaml_cli_agree_ok(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("intervention_encoding: layerwise\n")
    a = _args(model_config=str(p), intervention_encoding="layerwise")
    apply_model_config(a)
    assert a.intervention_encoding == "layerwise"


def test_unknown_yaml_key_raises(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("not_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown model_config key"):
        apply_model_config(_args(model_config=str(p)))


def test_existing_layerwise_config_still_resolves():
    a = _args(model_config=str(CONFIGS / "oracle_d3_model_m_layerwise.yaml"))
    apply_model_config(a)
    cls, name = resolve_model_class(a.intervention_encoding, a.model_class)
    assert cls is LayerwiseMixedContextMoGModel and name == "layerwise"


@pytest.mark.parametrize(
    "extra, cls",
    [
        ({}, MixedContextMoGModel),
        ({"intervention_encoding": "layerwise"}, LayerwiseMixedContextMoGModel),
        ({"model_class": "LayerwiseMixedContextMoGModel"}, LayerwiseMixedContextMoGModel),
    ],
)
def test_load_model_uses_registry(tmp_path, extra, cls):
    import yaml

    c = tiny_config()
    cfg = {
        "d_model": c.d_model,
        "emb_depth": c.emb_depth,
        "dim_feedforward": c.dim_feedforward,
        "nhead": c.nhead,
        "num_layers_encoder": c.num_layers_encoder,
        "sample_attn_mode": c.sample_attn_mode,
        "num_mixture_components": c.num_mixture_components,
        **extra,
    }
    yml = tmp_path / "model.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    torch.manual_seed(0)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": cls(c).state_dict()}, ckpt)
    model = load_model(str(ckpt), str(yml), "cpu", num_nodes=c.num_nodes)
    assert type(model) is cls