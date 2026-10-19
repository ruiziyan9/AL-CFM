"""Tests for the layer-wise flag injection (LayerwiseMixedContextMoGModel)."""

import sys
from pathlib import Path

import pytest
import torch

from CITNP.models.mixed_context_model import (
    LAYERWISE_PARAM_PREFIXES,
    InjectedCausalInfTransformerEncoder,
    LayerwiseMixedContextMoGModel,
    MixedContextMoGModel,
    load_r1_checkpoint,
    load_upstream_checkpoint,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mixed_context_model import (  # noqa: E402
    NC,
    B,
    D,
    inputs,
    outputs,
    randomise_flag,
    tiny_config,
)


def models(mode="MHCA"):
    cfg = tiny_config()
    cfg.sample_attn_mode = mode
    torch.manual_seed(0)
    r1 = MixedContextMoGModel(cfg).eval()
    randomise_flag(r1)  # a "trained" R1 embedding flag
    lw = LayerwiseMixedContextMoGModel(cfg).eval()
    load_r1_checkpoint(lw, r1.state_dict())
    return r1, lw


def randomise_injection(lw, seed=11):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in lw.named_parameters():
            if name.startswith(LAYERWISE_PARAM_PREFIXES):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def test_structure():
    _, lw = models()
    assert isinstance(lw.encoder.encoder, InjectedCausalInfTransformerEncoder)
    n_blocks = len(lw.encoder.encoder.iterator)
    assert len(lw.encoder.encoder.inject_sample) == n_blocks
    assert len(lw.encoder.encoder.inject_node) == n_blocks - 1


@pytest.mark.parametrize("mode", ["MHCA", "MHSA"])
@pytest.mark.parametrize("mask_name", ["none", "random", "all"])
def test_zero_init_equals_r1(mode, mask_name):
    r1, lw = models(mode)
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        a = outputs(r1, ctx, tgt, T, Y, masks[mask_name])
        b = outputs(lw, ctx, tgt, T, Y, masks[mask_name])
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("mode", ["MHCA", "MHSA"])
def test_injection_only_acts_on_flagged_cells(mode):
    r1, lw = models(mode)
    randomise_injection(lw)
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        # no flags anywhere -> identical to R1 (and to upstream)
        assert torch.allclose(
            outputs(r1, ctx, tgt, T, Y, masks["none"]),
            outputs(lw, ctx, tgt, T, Y, masks["none"]),
            atol=1e-6,
        )
        # one flagged cell -> the injection changes the prediction
        one = masks["none"].clone()
        one[:, 0, 0] = True
        assert not torch.allclose(
            outputs(r1, ctx, tgt, T, Y, one),
            outputs(lw, ctx, tgt, T, Y, one),
            atol=1e-4,
        )


def test_injection_gets_gradient_at_init():
    """Zero-init maps must still learn: flag_code is non-zero from step 0."""
    _, lw = models()
    lw.train()
    ctx, tgt, T, Y, masks = inputs()
    outputs(lw, ctx, tgt, T, Y, masks["random"]).sum().backward()
    for lin in (
        *lw.encoder.encoder.inject_sample,
        *lw.encoder.encoder.inject_node,
    ):
        assert lin.weight.grad is not None
        assert lin.weight.grad.abs().sum() > 0


def test_context_order_invariance_and_no_leakage():
    _, lw = models()
    randomise_injection(lw)
    ctx, tgt, T, Y, masks = inputs()
    perm = torch.randperm(NC, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = outputs(lw, ctx, tgt, T, Y, masks["random"])
        b = outputs(lw, ctx[:, perm], tgt, T, Y, masks["random"][:, perm])
    assert torch.allclose(a, b, atol=1e-5)
    tgt = tgt.clone().requires_grad_(True)
    outputs(lw, ctx, tgt, T, Y, masks["random"]).sum().backward()
    grad = tgt.grad[..., 0]
    for b_ in range(B):
        non_treat = [d for d in range(D) if d != int(T[b_])]
        assert (grad[b_][:, non_treat] == 0).all()


def test_checkpoint_loaders():
    r1, lw = models()
    # R1 -> Layerwise: only the layer-wise parameters are missing
    res = load_r1_checkpoint(
        LayerwiseMixedContextMoGModel(tiny_config()), r1.state_dict()
    )
    assert res.missing_keys and all(
        k.startswith(LAYERWISE_PARAM_PREFIXES) for k in res.missing_keys
    )
    # upstream-style (no context_intervened either) -> Layerwise also works
    up_sd = {
        k: v
        for k, v in r1.state_dict().items()
        if not k.startswith("encoder.context_intervened.")
    }
    load_upstream_checkpoint(LayerwiseMixedContextMoGModel(tiny_config()), up_sd)
    # a full Layerwise checkpoint is not an R1 checkpoint
    with pytest.raises(RuntimeError, match="mismatch"):
        load_r1_checkpoint(
            LayerwiseMixedContextMoGModel(tiny_config()), lw.state_dict()
        )
