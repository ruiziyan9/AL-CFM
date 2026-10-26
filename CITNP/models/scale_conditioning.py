"""Scale conditioning (scale_conditioning: stats).

Per-task standardisation removes the scale of every variable from the
context. The exact oracle sees the raw context, so for information parity the
model receives the removed statistics as well: per variable j the pair
(log std_j, mean_j / std_j), from which the raw context is recovered as

    x_raw = x_std * exp(log std_j) + (mean_j / std_j) * std_j.

`ScaleConditioning` maps that pair to a [d_model] embedding that is added to
every row (context and target) of variable column j, right after the row
embeddings are assembled and before the transformer stack. Its last Linear
layer is zero-initialised, so a fresh model computes the same function as one
without scale conditioning.
"""

import torch
import torch.nn as nn
from torch import Tensor

from CITNP.models.nncomponents import build_mlp

SCALE_CONDITIONING_OPTIONS = ("none", "stats")
# Parameters that checkpoints without scale conditioning do not have.
SCALE_PARAM_PREFIX = "encoder.scale_embedding."


def check_scale_conditioning(name: str) -> str:
    """Return `name` if it is a valid scale_conditioning option, else raise."""
    if name not in SCALE_CONDITIONING_OPTIONS:
        raise ValueError(
            f"Unknown scale_conditioning {name!r}; "
            f"choose from {SCALE_CONDITIONING_OPTIONS}"
        )
    return name


class ScaleConditioning(nn.Module):
    """MLP 2 -> d_model (depth `emb_depth`) of the per-variable standardisation
    statistics, last Linear layer zero-initialised."""

    def __init__(self, d_model: int, emb_depth: int = 1):
        super().__init__()
        self.mlp = build_mlp(dim_in=2, dim_hid=d_model, dim_out=d_model, depth=emb_depth)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, context_scale: Tensor) -> Tensor:
        """[B, D, 2] (log std, mean / std) -> [B, D, d_model]."""
        return self.mlp(context_scale)

    def add_to(self, full_data: Tensor, context_scale: Tensor) -> Tensor:
        """Add the embedding of variable j to every row of column j of
        `full_data` [B, N, D', d_model], D' >= D. Columns beyond the D
        variables (e.g. the extra-node metadata column) get nothing."""
        emb = self(context_scale.to(full_data.dtype))  # [B, D, d_model]
        D = emb.shape[1]
        if D > full_data.shape[2]:
            raise ValueError(f"context_scale has {D} variables, data {full_data.shape[2]}")
        if D < full_data.shape[2]:
            pad = emb.new_zeros(emb.shape[0], full_data.shape[2] - D, emb.shape[2])
            emb = torch.cat([emb, pad], dim=1)
        return full_data + emb[:, None]
