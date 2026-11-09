"""Record `context_states` and `predict_mixture` outputs BEFORE they are
extended for the flag-propagation diagnostic.

Run once, from the repository root, on the commit that adds this script
(before any edit to model_api.py):

    PYTHONPATH=. python testing/make_model_api_states_golden.py

Writes testing/fixtures/model_api_states_golden.pt: for every architecture in
flag_propagation_spec.ARCHITECTURES and both sample-attention modes, the
seeded tiny model's state dict (float32) and its default `context_states`
and `predict_mixture` outputs (with and without zero_flags) on one context.
CPU; deterministic.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flag_propagation_spec import ARCHITECTURES, build_model, regime_context  # noqa: E402

from CITNP.evaluation.model_api import context_states, predict_mixture  # noqa: E402

OUT = HERE / "fixtures" / "model_api_states_golden.pt"
T, Y = 0, 2
GRID = np.linspace(-1.5, 1.5, 4)


def record() -> dict:
    x, ci, _ = regime_context(seed=3)
    out = {"ctx_x": x, "ctx_int": ci, "T": T, "Y": Y, "grid": GRID, "models": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for arch in ARCHITECTURES:
            for mode in ("MHCA", "MHSA"):
                model = build_model(Path(tmp), arch, mode=mode, dtype=torch.float32)
                rec = {"state_dict": model.state_dict()}
                for zero in (False, True):
                    rec[f"states_zero{int(zero)}"] = context_states(
                        model, x, ci, T, Y, device="cpu", zero_flags=zero
                    )
                    rec[f"mixture_zero{int(zero)}"] = predict_mixture(
                        model, x, ci, T, Y, GRID, device="cpu", zero_flags=zero
                    )
                out["models"][f"{arch}/{mode}"] = rec
    return out


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    torch.save(record(), OUT)
    print(f"wrote {OUT}")
