"""Record golden outputs of the legacy encodings BEFORE reg_axis is added.

Run once, from the repository root, on the commit that adds this script
(before any reg_axis integration edit):

    PYTHONPATH=. python testing/make_reg_axis_legacy_golden.py

Writes testing/fixtures/reg_axis_legacy_golden.pt with, for each case
(r1, layerwise, extra_node) x (scale_conditioning none, stats):

    state dict (float16 storage, see below), inputs, and the outputs
    pred_mean / pred_std / weights of the eval-mode model.

Every parameter is perturbed with seeded noise, so the zero-initialised flag,
injection and scale-embedding layers all affect the outputs. Parameters are
then rounded to float16 and back, so storing them as float16 is lossless and
keeps the file small. Each "stats" model copies every parameter of the
"none" model of the same encoding, so only its `encoder.scale_embedding.*`
keys are stored; the test merges the two dicts and loads strictly.

Runs on CPU; deterministic.
"""

import sys
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_mixed_context_model import inputs, tiny_config  # noqa: E402

from CITNP.models.mixed_context_model import INTERVENTION_ENCODINGS  # noqa: E402

OUT = HERE / "fixtures" / "reg_axis_legacy_golden.pt"
METHODS = ("r1", "layerwise", "extra_node")
SCALES = ("none", "stats")
SCALE_PREFIX = "encoder.scale_embedding."


def build(method: str, scale: str, base: dict | None = None) -> torch.nn.Module:
    """Seeded tiny model with every parameter perturbed and float16-exact;
    parameters present in `base` are copied from it instead."""
    cfg = tiny_config()
    cfg.scale_conditioning = scale
    torch.manual_seed(0)
    model = INTERVENTION_ENCODINGS[method](cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            # Seeded by name, so shared parameters get the same noise with and
            # without scale conditioning.
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            p.add_(0.1 * torch.randn(p.shape, generator=g))
            p.copy_(p.half().float())
        if base is not None:
            missing = model.load_state_dict(base, strict=False).missing_keys
            assert all(k.startswith(SCALE_PREFIX) for k in missing), missing
    return model


def pack(sd: dict) -> dict:
    """One flat float16 tensor plus (name, shape) pairs: far smaller on disk
    than one zip entry per tensor. Inverse: test_reg_axis_legacy_golden.unpack."""
    return {
        "keys": [(k, tuple(v.shape)) for k, v in sd.items()],
        "flat": torch.cat([v.reshape(-1) for v in sd.values()]).half(),
    }


def golden_inputs() -> dict:
    ctx, tgt, T, Y, masks = inputs(seed=1)
    g = torch.Generator().manual_seed(5)
    B, _, D, _ = ctx.shape
    scale = torch.stack(
        [0.5 * torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)],
        dim=-1,
    )
    return {
        "context_data": ctx,
        "target_data": tgt,
        "intervention_index": T,
        "outcome_index": Y,
        "context_int_mask": masks["random"],
        "context_scale": scale,
    }


def run(model: torch.nn.Module, inp: dict) -> dict:
    with torch.no_grad():
        out = model(**inp)
    return {
        "pred_mean": out.pred_mean,
        "pred_std": out.pred_std,
        "weights": out.weights,
    }


def main() -> None:
    inp = golden_inputs()
    cases = {}
    for method in METHODS:
        base = None
        for scale in SCALES:
            model = build(method, scale, base)
            sd = model.state_dict()
            if scale == "none":
                base = sd
                stored = sd
            else:
                assert base is not None
                for k, v in base.items():
                    assert torch.equal(sd[k], v), (method, k)
                stored = {k: v for k, v in sd.items() if k.startswith(SCALE_PREFIX)}
                assert set(sd) == set(base) | set(stored)
            cases[f"{method}/{scale}"] = {
                "state_dict": pack(stored),
                "outputs": run(model, inp),
            }
    OUT.parent.mkdir(exist_ok=True)
    torch.save({"inputs": inp, "cases": cases}, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
