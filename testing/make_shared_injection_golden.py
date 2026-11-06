"""Record golden outputs of the default layerwise model BEFORE
share_intervention_encoder is added.

Run once, from the repository root, on the commit that adds this script
(before any edit to the layerwise model):

    PYTHONPATH=. python testing/make_shared_injection_golden.py

Writes testing/fixtures/shared_injection_golden.pt with
- param_names: the default layerwise model's parameter names, in order;
- state_dict: its seeded (torch.manual_seed(0)) initial state dict;
- inputs: a fixed batch with unflagged context cells, flagged cells whose
  value is exactly 0, flagged cells with nonzero values, and query rows;
- outputs_init: eval-mode outputs of the seeded model;
- outputs_perturbed: eval-mode outputs after `perturb` (every parameter
  moved by seeded noise, so the zero-initialised flag and injection layers
  all affect the prediction).

Runs on CPU; deterministic.
"""

import sys
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_mixed_context_model import inputs, tiny_config  # noqa: E402

from CITNP.models.mixed_context_model import (  # noqa: E402
    LayerwiseMixedContextMoGModel,
)

OUT = HERE / "fixtures" / "shared_injection_golden.pt"


def perturb(model: torch.nn.Module) -> torch.nn.Module:
    """Add noise seeded by parameter name to every parameter (in place)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return model


def golden_inputs() -> dict:
    """Random context with ~30% flagged cells; a third of the flagged cells
    are set to exactly 0, and so is one unflagged cell per batch element."""
    ctx, tgt, T, Y, masks = inputs(seed=1)
    mask = masks["random"].clone()
    mask[:, 0, 0] = True  # guarantee flagged cells of both kinds
    mask[:, 1, 1] = True
    mask[:, 2, 2] = False
    ctx = ctx.clone()
    g = torch.Generator().manual_seed(9)
    zero = mask & (torch.rand(mask.shape, generator=g) < 1 / 3)
    zero[:, 0, 0] = True
    zero[:, 1, 1] = False
    ctx[zero] = 0.0
    ctx[:, 2, 2] = 0.0  # unflagged cell with value 0
    return {
        "context_data": ctx,
        "target_data": tgt,
        "intervention_index": T,
        "outcome_index": Y,
        "context_int_mask": mask,
    }


def build() -> torch.nn.Module:
    torch.manual_seed(0)
    return LayerwiseMixedContextMoGModel(tiny_config()).eval()


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
    model = build()
    fixture = {
        "param_names": [n for n, _ in model.named_parameters()],
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "inputs": inp,
        "outputs_init": run(model, inp),
        "outputs_perturbed": run(perturb(model), inp),
    }
    OUT.parent.mkdir(exist_ok=True)
    torch.save(fixture, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
