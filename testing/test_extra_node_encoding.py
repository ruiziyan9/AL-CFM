"""Tests for the extra-node intervention encoding
(ExtraNodeMixedContextMoGModel, CITNP/models/extra_node_encoder.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from CITNP.evaluation.model_api import context_states, predict_mixture
from CITNP.models.causalinferencemodel import MoGCausalInferenceModel
from CITNP.models.mixed_context_model import (
    EXTRA_NODE_PARAM_PREFIXES,
    NEW_PARAM_PREFIX,
    ExtraNodeMixedContextMoGModel,
    MixedContextModel,
    load_upstream_checkpoint,
    resolve_model_class,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import (  # noqa: E402
    NC,
    NQ,
    B,
    D,
    inputs,
    outputs,
    tiny_config,
)

MODES = ["MHCA", "MHSA"]


def model(mode="MHCA", num_nodes=D, seed=0):
    cfg = tiny_config()
    cfg.sample_attn_mode = mode
    cfg.num_nodes = num_nodes
    torch.manual_seed(seed)
    return ExtraNodeMixedContextMoGModel(cfg).eval()


def randomise_embeddings(m, seed=7):
    """Make the index / metadata embeddings large enough to matter."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.startswith(EXTRA_NODE_PARAM_PREFIXES):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_registry_resolves():
    cls, name = resolve_model_class("extra_node")
    assert cls is ExtraNodeMixedContextMoGModel and name == "extra_node"
    cls, name = resolve_model_class("extra_node", "ExtraNodeMixedContextMoGModel")
    assert cls is ExtraNodeMixedContextMoGModel and name == "extra_node"


def test_extra_node_is_not_an_r1_model():
    assert not isinstance(model(), MixedContextModel)


def test_parameters():
    keys = list(model().state_dict())
    assert not [k for k in keys if k.startswith(NEW_PARAM_PREFIX)]
    assert not [k for k in keys if "flag_code" in k or "inject_" in k]
    assert len([k for k in keys if k.startswith(EXTRA_NODE_PARAM_PREFIXES)]) == 2


def test_context_metadata_values():
    enc = model().encoder
    mask = torch.tensor(
        [
            [
                [False, False, False],  # observational
                [False, False, True],  # do(X_2)
                [True, False, True],  # do(X_0, X_2)
            ]
        ]
    )
    with torch.no_grad():
        meta = enc.context_metadata(mask, 1, 3, D)
        E = enc.index_embedding.weight
        e_int = enc.int_meta_embedding
    assert meta.shape == (1, 3, 1, E.shape[1])
    torch.testing.assert_close(meta[0, 0, 0], e_int)
    torch.testing.assert_close(meta[0, 1, 0], e_int + E[2])
    torch.testing.assert_close(meta[0, 2, 0], e_int + E[0] + E[2])


def test_query_metadata_is_independent_of_query_data():
    enc = model().encoder
    g = torch.Generator().manual_seed(3)
    q = torch.randn(B, NQ, D, 1, generator=g)
    with torch.no_grad():
        got = enc.query_metadata(q)
        want = enc.target_variable(torch.zeros(1, 1, 1, 1))
    torch.testing.assert_close(got, want.expand(B, NQ, 1, -1))


def test_mask_semantics():
    m = model()
    randomise_embeddings(m)
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        none = outputs(m, ctx, tgt, T, Y)
        zero = outputs(m, ctx, tgt, T, Y, masks["none"])
        rand = outputs(m, ctx, tgt, T, Y, masks["random"])
    torch.testing.assert_close(none, zero)
    assert not torch.allclose(zero, rand, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_encode_shapes(mode):
    m = model(mode)
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        rep, num_target = m.encoder.encode(
            context_data=ctx,
            target_data=m._mask_target_data(tgt, T),
            outcome_indices=Y,
            intervention_indices=T,
            context_int_mask=masks["random"],
        )
    assert rep.shape == (B, NC + NQ, D, m.d_model)
    assert num_target == NQ


@pytest.mark.parametrize("mode", MODES)
def test_no_leakage_from_masked_query_cells(mode):
    m = model(mode)
    randomise_embeddings(m)
    ctx, tgt, T, Y, masks = inputs()
    is_treatment = torch.nn.functional.one_hot(T, D).bool()[:, None, :, None]
    shifted = torch.where(is_treatment, tgt, tgt + 5.0)
    with torch.no_grad():
        ref = outputs(m, ctx, tgt, T, Y, masks["random"])
        got = outputs(m, ctx, shifted, T, Y, masks["random"])
    torch.testing.assert_close(ref, got, atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_query_rows_do_not_see_each_other(mode):
    m = model(mode)
    randomise_embeddings(m)
    ctx, tgt, T, Y, masks = inputs()
    bumped = tgt.clone()
    bumped[torch.arange(B), 0, T] += 1.0  # only query row 0's do-value
    with torch.no_grad():
        ref = outputs(m, ctx, tgt, T, Y, masks["random"])
        got = outputs(m, ctx, bumped, T, Y, masks["random"])
    torch.testing.assert_close(ref[:, 1:], got[:, 1:], atol=1e-6, rtol=0)
    assert not torch.allclose(ref[:, 0], got[:, 0], atol=1e-6)


def test_upstream_warm_start():
    up = MoGCausalInferenceModel(tiny_config()).eval()
    m = model()
    result = load_upstream_checkpoint(m, up.state_dict())
    expected = {
        k for k in m.state_dict() if k.startswith(EXTRA_NODE_PARAM_PREFIXES)
    }
    assert len(expected) == 2
    assert set(result.missing_keys) == expected
    with pytest.raises(RuntimeError, match="Unexpected"):
        load_upstream_checkpoint(
            m, {**up.state_dict(), "encoder.bogus": torch.zeros(1)}
        )


def test_variable_count():
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        out = outputs(model(num_nodes=D + 2), ctx, tgt, T, Y, masks["random"])
    assert out.shape[:2] == (B, NQ)
    with pytest.raises(ValueError, match="at most"):
        with torch.no_grad():
            outputs(model(num_nodes=D - 1), ctx, tgt, T, Y, masks["random"])


def test_model_api():
    m = model()
    randomise_embeddings(m)
    rng = np.random.default_rng(0)
    ctx_x = rng.normal(size=(NC, D)).astype(np.float32)
    ctx_int = np.zeros((NC, D), dtype=bool)
    ctx_int[NC // 2 :, 1] = True
    t = np.linspace(-1.0, 1.0, 5)
    K = m.predictor.num_mixture_components
    for zero_flags in (False, True):
        mu, sd, logw = predict_mixture(
            m, ctx_x, ctx_int, 1, 2, t, device="cpu", zero_flags=zero_flags
        )
        assert mu.shape == (1, len(t), K)
    st = context_states(m, ctx_x, ctx_int, 1, 2, device="cpu")
    assert st
    for v in st.values():
        assert v.shape == (NC, D, m.d_model)
