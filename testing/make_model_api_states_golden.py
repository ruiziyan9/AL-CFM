"""Record `context_states` and `predict_mixture` outputs of the model_api code
as it was BEFORE the flag-propagation extension.

Run from the repository root with the pre-extension code first on the path,
e.g. from a worktree of the commit before `[flag-propagation 2]`:

    git worktree add /tmp/base 1bf8fa9
    PYTHONPATH=/tmp/base python testing/make_model_api_states_golden.py

Writes testing/fixtures/model_api_states_golden.pt: for every architecture in
flag_propagation_spec.ARCHITECTURES and both sample-attention modes, a
parameter checksum of the seeded tiny float64 model and its default
`context_states` (float32 by contract) and `predict_mixture` (float64)
outputs, with and without zero_flags, on one context. Also records the torch
version, device and dtype. CPU; deterministic.
"""

import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flag_propagation_spec import (  # noqa: E402
    ARCHITECTURES,
    GOLDEN_MODEL,
    build_model,
    parameter_checksum,
    regime_context,
)

import CITNP  # noqa: E402
from CITNP.evaluation.model_api import context_states, predict_mixture  # noqa: E402

OUT = HERE / "fixtures" / "model_api_states_golden.pt"
T, Y = 0, 2
GRID = np.linspace(-1.5, 1.5, 4)
DTYPE = torch.float64


def record() -> dict:
    x, ci, _ = regime_context(seed=3, n_obs=10, sizes=(1, 1))
    out = {
        "ctx_x": x, "ctx_int": ci, "T": T, "Y": Y, "grid": GRID,
        "environment": dict(
            torch_version=torch.__version__, device="cpu", dtype=str(DTYPE),
            numpy_version=np.__version__, platform=platform.platform(),
            citnp_path=str(Path(CITNP.__file__).parent),
        ),
        "models": {},
    }  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        for arch in ARCHITECTURES:
            for mode in ("MHCA", "MHSA"):
                model = build_model(
                    Path(tmp), arch, mode=mode, dtype=DTYPE, **GOLDEN_MODEL
                )
                rec = {"parameter_checksum": parameter_checksum(model)}
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
    rec = record()
    torch.save(rec, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB) with "
          f"{rec['environment']}")  # fmt: skip
