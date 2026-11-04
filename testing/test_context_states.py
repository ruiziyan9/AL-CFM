"""Tests for model_api.context_states / probe_points on every intervention
encoding, including the extra-node metadata column (random tiny models, CPU).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from CITNP.evaluation.model_api import (
    context_states,
    predict_mixture,
    probe_points,
    uses_metadata_node,
)
from CITNP.models.mixed_context_model import (
    INTERVENTION_ENCODINGS,
    resolve_model_class,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import D, tiny_config  # noqa: E402

NUM_CONTEXT_ROWS = 40
ENCODINGS = sorted(INTERVENTION_ENCODINGS)


def tiny_model(encoding: str, seed: int = 0) -> torch.nn.Module:
    """Random-initialised tiny model of one intervention encoding."""
    model_class, _ = resolve_model_class(encoding)
    torch.manual_seed(seed)
    return model_class(tiny_config()).eval()


def context(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Half observational rows, half rows intervened on one random column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(NUM_CONTEXT_ROWS, D)).astype(np.float32)
    intervened = np.zeros((NUM_CONTEXT_ROWS, D), dtype=bool)
    intervened[NUM_CONTEXT_ROWS // 2 :, rng.integers(D)] = True
    return x, intervened


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_default_states_have_one_column_per_variable(encoding):
    model = tiny_model(encoding)
    x, intervened = context()
    states = context_states(model, x, intervened, 0, 2, device="cpu")
    assert list(states) == probe_points(model)
    assert probe_points(model)[0] == "embed"
    for value in states.values():
        assert value.shape == (NUM_CONTEXT_ROWS, D, model.d_model)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_hooks_do_not_change_predictions(encoding):
    model = tiny_model(encoding)
    x, intervened = context()
    grid = np.linspace(-2, 2, 5)
    before = predict_mixture(model, x, intervened, 0, 2, grid, device="cpu")
    context_states(model, x, intervened, 0, 2, device="cpu")
    after = predict_mixture(model, x, intervened, 0, 2, grid, device="cpu")
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_regime_axis_probe_points():
    model = tiny_model("reg_axis")
    names = probe_points(model)
    num_blocks = len(model.encoder.blocks)
    # feedback in every block but the last
    assert len(names) == 1 + num_blocks + (num_blocks - 1)
    assert names[-1] == f"block{num_blocks - 1}/sample"


def test_extra_node_metadata_column():
    model = tiny_model("extra_node")
    assert uses_metadata_node(model)
    x, intervened = context()
    default = context_states(model, x, intervened, 0, 2, device="cpu")
    with_metadata = context_states(
        model, x, intervened, 0, 2, device="cpu", include_metadata_node=True
    )
    assert list(with_metadata) == list(default)
    for name, value in with_metadata.items():
        assert value.shape == (NUM_CONTEXT_ROWS, D + 1, model.d_model)
        assert torch.equal(value[:, :D], default[name])


def test_extra_node_metadata_input_is_the_documented_embedding():
    """At the encoder input, the metadata column is e_INT + sum_{j in S} e_j."""
    model = tiny_model("extra_node")
    x, intervened = context()
    states = context_states(
        model, x, intervened, 0, 2, device="cpu", include_metadata_node=True
    )
    encoder = model.encoder
    index_embeddings = encoder.index_embeddings(D).detach()
    expected = encoder.int_meta_embedding.detach() + torch.as_tensor(
        intervened, dtype=torch.float32
    ) @ index_embeddings
    torch.testing.assert_close(states["embed"][:, D], expected)


@pytest.mark.parametrize("encoding", [e for e in ENCODINGS if e != "extra_node"])
def test_metadata_column_on_other_encodings_raises(encoding):
    model = tiny_model(encoding)
    assert not uses_metadata_node(model)
    x, intervened = context()
    with pytest.raises(ValueError, match="extra_node"):
        context_states(
            model, x, intervened, 0, 2, device="cpu", include_metadata_node=True
        )
