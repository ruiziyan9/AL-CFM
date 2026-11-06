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
from test_mixed_context_model import (
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
        diff = outputs(r1, ctx, tgt, T, Y, one) - outputs(lw, ctx, tgt, T, Y, one)
        assert diff.abs().max() > 1e-6  # float32 noise is ~1e-7 here


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


# ---------------------------------------------------------------------------
# share_intervention_encoder: the injection reuses context_intervened.
# ---------------------------------------------------------------------------

import yaml  # type: ignore[import-untyped]  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

from CITNP.evaluation.model_api import load_model  # noqa: E402
from make_shared_injection_golden import (  # noqa: E402
    OUT as GOLDEN_PATH,
    build as build_golden,
    perturb,
)

# Parameter count of flag_code for tiny_config (d_model 32, emb_depth 2):
# Linear(1, 32) + Linear(32, 32) = (32 + 32) + (32 * 32 + 32) = 1120.
FLAG_CODE_PARAMS_TINY = 1120


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN_PATH, map_location="cpu")


def shared_cfg(share=True, mode="MHCA"):
    cfg = tiny_config()
    cfg.sample_attn_mode = mode
    cfg.share_intervention_encoder = share
    return cfg


def build_shared(share=True, seed=0):
    torch.manual_seed(seed)
    return LayerwiseMixedContextMoGModel(shared_cfg(share))


def projections(model):
    enc = model.encoder.encoder
    return [*enc.inject_sample, *enc.inject_node]


def intervened_linears(model):
    return [
        m
        for m in model.encoder.context_intervened.modules()
        if isinstance(m, torch.nn.Linear)
    ]


def capture_transformer_inputs(model):
    """Record src and flag_code as passed to the injection transformer."""
    seen = {}

    def hook(_mod, args, kwargs):
        seen["src"] = args[0] if args else kwargs["src"]
        seen["flag_code"] = kwargs.get("flag_code")

    return seen, model.encoder.encoder.register_forward_pre_hook(
        hook, with_kwargs=True
    )


def tiny_model_yaml(path, share=None):
    c = tiny_config()
    m = {
        "d_model": c.d_model,
        "emb_depth": c.emb_depth,
        "dim_feedforward": c.dim_feedforward,
        "nhead": c.nhead,
        "num_layers_encoder": c.num_layers_encoder,
        "sample_attn_mode": c.sample_attn_mode,
        "num_mixture_components": c.num_mixture_components,
        "intervention_encoding": "layerwise",
    }
    if share is not None:
        m["share_intervention_encoder"] = share
    path.write_text(yaml.safe_dump(m))
    return str(path)


# 1. Default behaviour ------------------------------------------------------


def test_default_matches_golden(golden):
    model = build_golden()
    assert not model.encoder.share_intervention_encoder
    assert [n for n, _ in model.named_parameters()] == golden["param_names"]
    sd = model.state_dict()
    assert list(sd) == list(golden["state_dict"])
    for k, v in golden["state_dict"].items():
        assert torch.equal(sd[k], v), k
    for key, prepare in (("outputs_init", None), ("outputs_perturbed", perturb)):
        if prepare is not None:
            prepare(model)
        with torch.no_grad():
            out = model(**golden["inputs"])
        for name, ref in golden[key].items():
            # Bit-exact in a fresh process on the machine that wrote the
            # fixture; CPU matmuls differ in the last bits with the thread
            # count and CPU, so compare at the legacy-golden tolerance.
            assert torch.allclose(getattr(out, name), ref, atol=1e-6, rtol=0), (
                key,
                name,
            )


def test_existing_layerwise_checkpoint_loads_strict(golden, tmp_path):
    """A checkpoint written before the option existed (no key in the YAML)
    rebuilds the default model and loads with strict=True."""
    ckpt = tmp_path / "last_model.pt"
    torch.save({"model_state_dict": golden["state_dict"]}, ckpt)
    model = load_model(str(ckpt), tiny_model_yaml(tmp_path / "m.yaml"), "cpu", D)
    assert hasattr(model.encoder, "flag_code")
    model.load_state_dict(golden["state_dict"], strict=True)
    with torch.no_grad():
        out = model(**golden["inputs"])
    for name, ref in golden["outputs_init"].items():
        assert torch.allclose(getattr(out, name), ref, atol=1e-6, rtol=0), name


# 2. Structure ----------------------------------------------------------------


def test_shared_structure():
    default, shared = build_shared(False), build_shared(True)
    assert not hasattr(shared.encoder, "flag_code")
    assert not any("flag_code" in n for n, _ in shared.named_parameters())
    n_default = sum(p.numel() for p in default.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    n_flag = sum(p.numel() for p in default.encoder.flag_code.parameters())
    assert n_flag == FLAG_CODE_PARAMS_TINY
    assert n_default - n_shared == FLAG_CODE_PARAMS_TINY
    # everything except flag_code (role MLPs, context_intervened, injection
    # projections, attention, head) has the same names and shapes
    shapes = lambda m: {  # noqa: E731
        n: p.shape
        for n, p in m.named_parameters()
        if not n.startswith("encoder.flag_code.")
    }
    assert shapes(default) == shapes(shared)
    assert any(n.startswith("encoder.encoder.inject_") for n in shapes(shared))


# 3. Same tensor ----------------------------------------------------------------


class _RecordAddCat(TorchFunctionMode):
    """Records the operands of every tensor addition and torch.cat."""

    def __init__(self):
        super().__init__()
        self.adds, self.cats = [], []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (torch.Tensor.__add__, torch.add, torch.Tensor.add):
            self.adds.append((args[0], args[1], out))
        elif func is torch.cat:
            self.cats.append((list(args[0]), out))
        return out


def _trace(model, mask_name="random"):
    """Run one forward pass; return the masked context_intervened output (as
    passed on for injection), every add / cat, and each projection's input."""
    ctx, tgt, T, Y, masks = inputs()
    calls, proj_in = [], []
    hooks = [
        model.encoder.context_intervened.register_forward_hook(
            lambda *a: calls.append(1)
        )
    ]
    hooks += [
        lin.register_forward_pre_hook(lambda _m, a: proj_in.append(a[0]))
        for lin in projections(model)
    ]
    passed = {}
    orig = model.encoder._extra_encoder_inputs

    def spy(context_data, context_int_mask, full_data, intervened_code=None):
        passed["code"] = intervened_code
        return orig(context_data, context_int_mask, full_data, intervened_code)

    model.encoder._extra_encoder_inputs = spy
    rec = _RecordAddCat()
    try:
        with rec:
            outputs(model, ctx, tgt, T, Y, masks[mask_name])
    finally:
        del model.encoder._extra_encoder_inputs
        for h in hooks:
            h.remove()
    return passed["code"], rec, proj_in, len(calls)


def test_shared_embedding_and_injection_use_same_tensor():
    model = build_shared(True)
    randomise_flag(model)
    code, rec, proj_in, n_calls = _trace(model)
    ptr = code.data_ptr()
    assert n_calls == 1  # context_intervened runs once per forward
    # the embedding addition (context_embed + code) receives this tensor
    emb_adds = [a for a in rec.adds if a[1].data_ptr() == ptr]
    assert len(emb_adds) == 1 and emb_adds[0][1] is code
    # the padding step receives the same tensor, unpadded, first
    pads = [c for c in rec.cats if c[0][0].data_ptr() == ptr]
    assert len(pads) == 1 and pads[0][0][0] is code
    padded = pads[0][1]
    # and every injection projection receives that padded tensor itself
    assert len(proj_in) == len(projections(model))
    assert all(x.data_ptr() == padded.data_ptr() for x in proj_in)
    assert torch.equal(padded[:, :NC], code)
    # autograd: the projections' input is built directly from the same node
    assert padded.grad_fn.next_functions[0][0] is code.grad_fn


def test_default_mode_uses_separate_code():
    model = build_shared(False)
    randomise_flag(model)
    code, rec, proj_in, _ = _trace(model)
    pads = [c for c in rec.cats if c[0][0].shape == code.shape]
    assert pads and all(c[0][0].data_ptr() != code.data_ptr() for c in pads)


# 4. Masking --------------------------------------------------------------------


def test_shared_code_masking(golden):
    model = build_shared(True).eval()
    randomise_flag(model)  # nonzero weights in context_intervened
    inp = golden["inputs"]
    mask, ctx = inp["context_int_mask"], inp["context_data"]
    seen, h = capture_transformer_inputs(model)
    with torch.no_grad():
        model(**inp)
    h.remove()
    code = seen["flag_code"]
    nc = ctx.shape[1]
    assert code.shape[1] > nc
    assert (code[:, nc:] == 0).all()  # query rows
    assert (code[:, :nc][~mask] == 0).all()  # unflagged context cells
    # a flagged cell with value exactly 0 differs from an unflagged cell
    zero_flagged = mask & (ctx[..., 0] == 0)
    zero_unflagged = ~mask & (ctx[..., 0] == 0)
    assert zero_flagged.any() and zero_unflagged.any()
    flagged_codes = code[:, :nc][zero_flagged]
    assert (flagged_codes.abs().amax(-1) > 0).all()
    assert not torch.equal(flagged_codes[0], code[:, :nc][zero_unflagged][0])


# 5. Gradients -------------------------------------------------------------------


def _weighted_loss(model, w):
    ctx, tgt, T, Y, masks = inputs()
    return (outputs(model, ctx, tgt, T, Y, masks["random"]) * w).sum()


def _loss_weights(model):
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        shape = outputs(model, ctx, tgt, T, Y, masks["random"]).shape
    return torch.randn(shape, generator=torch.Generator().manual_seed(4))


def test_shared_gradients_at_step_zero_and_after_updates():
    model = build_shared(True).train()
    w = _loss_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    *hidden, last = intervened_linears(model)
    assert hidden

    opt.zero_grad()
    _weighted_loss(model, w).backward()
    # step 0: final layer of context_intervened and the projections are zero
    for lin in projections(model):
        assert lin.weight.grad is not None and (lin.weight.grad == 0).all()
    assert last.weight.grad.abs().sum() > 0
    for lin in hidden:
        assert (lin.weight.grad == 0).all() and (lin.bias.grad == 0).all()

    # One update: the projections had zero gradient, so they are still zero,
    # but context_intervened's code is now nonzero and they receive gradient.
    opt.step()
    opt.zero_grad()
    _weighted_loss(model, w).backward()
    for lin in projections(model):
        assert (lin.weight == 0).all()
        assert lin.weight.grad.abs().sum() > 0
    # Second update: the projections become nonzero and keep receiving gradient.
    opt.step()
    opt.zero_grad()
    _weighted_loss(model, w).backward()
    for lin in projections(model):
        assert lin.weight.abs().sum() > 0
        assert lin.weight.grad.abs().sum() > 0


def test_shared_injection_path_contributes_gradient():
    def intervened_grad(proj_values):
        model = build_shared(True).train()
        w = _loss_weights(model)
        g = torch.Generator().manual_seed(21)
        with torch.no_grad():
            for lin in projections(model):
                lin.weight.copy_(
                    proj_values * torch.randn(lin.weight.shape, generator=g)
                )
        _weighted_loss(model, w).backward()
        return torch.cat(
            [
                p.grad.reshape(-1)
                for p in model.encoder.context_intervened.parameters()
            ]
        )

    at_zero = intervened_grad(0.0)
    nonzero = intervened_grad(0.1)
    assert not torch.allclose(at_zero, nonzero)


# 6. Smoke ----------------------------------------------------------------------


@pytest.mark.parametrize("share", [False, True])
def test_shared_smoke_train_save_reload(share, tmp_path):
    model = build_shared(share).train()
    ctx, tgt, T, Y, masks = inputs()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(4):
        opt.zero_grad()
        out = model(
            context_data=ctx,
            target_data=tgt,
            intervention_index=T,
            outcome_index=Y,
            context_int_mask=masks["random"],
        )
        loss = model.calculate_loss(out, target=tgt, outcome_index=Y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    ckpt = tmp_path / "last_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    # rebuilt from the YAML hyperparameters, loaded strictly
    yml = tiny_model_yaml(tmp_path / "m.yaml", share)
    reloaded = load_model(str(ckpt), yml, "cpu", D)
    assert reloaded.encoder.share_intervention_encoder is share
    model.eval()
    with torch.no_grad():
        a = outputs(model, ctx, tgt, T, Y, masks["random"])
        b = outputs(reloaded, ctx, tgt, T, Y, masks["random"])
    assert torch.equal(a, b)


def test_checkpoint_modes_do_not_cross_load():
    shared_sd = build_shared(True).state_dict()
    with pytest.raises(RuntimeError, match="flag_code"):
        build_shared(False).load_state_dict(shared_sd, strict=True)
    with pytest.raises(RuntimeError, match="flag_code"):
        build_shared(True).load_state_dict(
            build_shared(False).state_dict(), strict=True
        )


def test_r1_checkpoint_warm_starts_shared_model():
    r1, _ = models()
    shared = build_shared(True).eval()
    res = load_r1_checkpoint(shared, r1.state_dict())
    assert res.missing_keys and all(
        k.startswith("encoder.encoder.inject_") for k in res.missing_keys
    )
    ctx, tgt, T, Y, masks = inputs()
    with torch.no_grad():
        assert torch.allclose(
            outputs(r1, ctx, tgt, T, Y, masks["random"]),
            outputs(shared, ctx, tgt, T, Y, masks["random"]),
            atol=1e-6,
        )


# Config ------------------------------------------------------------------------

CONFIGS = (
    Path(__file__).resolve().parents[1] / "CITNP" / "experiments" / "configs"
)


def test_shared_config_differs_from_base_only_in_option():
    base = yaml.safe_load(
        (CONFIGS / "oracle_d3_model_m_layerwise_stats.yaml").read_text()
    )
    shared = yaml.safe_load(
        (CONFIGS / "oracle_d3_model_m_layerwise_stats_shared.yaml").read_text()
    )
    assert shared.pop("share_intervention_encoder") is True
    assert "share_intervention_encoder" not in base
    assert shared == base


def _train_args(tmp_path, model_config):
    from CITNP.experiments.mixed_train import parse_args

    return parse_args(
        [
            "--data_config", str(CONFIGS / "smoke_d3.yaml"),
            "--model_config", str(model_config),
            "--device", "cpu", "--dtype", "float32",
            "--batch_size", "4", "--iterations_per_epoch", "2",
            "--epochs", "1", "--val_iterations", "1", "--test_iterations", "1",
            "--num_workers", "0", "--no_wandb", "--seed", "0",
            "--save_checkpoint_every_n_steps", "1000",
            "--results_path", str(tmp_path), "--run_name", "r",
        ]
    )  # fmt: skip


def _smoke_yaml(tmp_path, **extra):
    m = yaml.safe_load((CONFIGS / "smoke_d3_model.yaml").read_text())
    m.update(extra)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(m))
    return p


@pytest.mark.parametrize("share", [None, False, True])
def test_yaml_to_model_to_resolved_config(tmp_path, share):
    """YAML key -> constructor -> resolved_model_config.yaml -> load_model."""
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.experiments.mixed_train import build_trainer

    extra = {"intervention_encoding": "layerwise"}
    if share is not None:
        extra["share_intervention_encoder"] = share
    data_cfg = load_config(CONFIGS / "smoke_d3.yaml")
    trainer, _ = build_trainer(
        _train_args(tmp_path, _smoke_yaml(tmp_path, **extra)), data_cfg
    )
    expected = bool(share)  # missing key => False
    assert trainer.model.encoder.share_intervention_encoder is expected
    assert hasattr(trainer.model.encoder, "flag_code") is not expected
    trainer.train()
    run = tmp_path / "results" / "r"
    resolved = yaml.safe_load((run / "resolved_model_config.yaml").read_text())
    assert resolved["share_intervention_encoder"] is expected
    model = load_model(
        str(run / "last_model.pt"), str(run / "resolved_model_config.yaml"), "cpu"
    )
    assert model.encoder.share_intervention_encoder is expected


def test_share_option_needs_layerwise(tmp_path):
    from CITNP.datasets.create_mixed_regime_chunks import load_config
    from CITNP.experiments.mixed_train import build_trainer

    yml = _smoke_yaml(tmp_path, share_intervention_encoder=True)  # r1 default
    with pytest.raises(ValueError, match="only applies to the layerwise"):
        build_trainer(
            _train_args(tmp_path, yml), load_config(CONFIGS / "smoke_d3.yaml")
        )
