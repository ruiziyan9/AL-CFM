"""Intervention-target flag for reg_axis context cells.

Terminology: an "intervention target" is a context cell set by an
experiment (context_int_mask True). It is unrelated to the "treatment role",
the column of the query's T, which the role MLPs (`*_intervention`) mark.

Mode "injected" (default) mirrors the functional form of the layer-wise
encoding with its own parameters:

    h0   += flag * flag_embed(x)                (once, at the input)
    code  = flag * flag_code(x)                 (zero off intervened cells)
    before steps 1 and 2 of block l:
    hc   += inject_var[l](code)  /  inject_sample[l](code)

flag_embed's last layer and the bias-free inject maps start at zero, so
non-intervened cells are never touched and the flag starts with no effect.
Query rows never get a flag. Mode "none" has no parameters at all (regimes
are then known only through the grouping).
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from CITNP.models.nncomponents import build_mlp

FLAG_MODES = ("injected", "none")


def _zero_last_linear(mlp: nn.Module) -> None:
    last = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class FlagInjection(nn.Module):
    """Flag embedding and per-block injection of intervened context cells.

    Shapes: x [B, Nc, D, 1] context values, flag [B, Nc, D] bool,
    embeddings / codes [B, Nc, D, d].
    """

    def __init__(self, d_model: int, emb_depth: int, num_blocks: int, mode: str):
        super().__init__()
        if mode not in FLAG_MODES:
            raise ValueError(
                f"Unknown reg_axis_target_flag {mode!r}; choose from {FLAG_MODES}"
            )
        self.mode = mode
        if mode == "none":
            return
        mlp = dict(dim_in=1, dim_hid=d_model, dim_out=d_model, depth=emb_depth)
        self.flag_embed = build_mlp(**mlp)
        _zero_last_linear(self.flag_embed)
        self.flag_code = build_mlp(**mlp)
        self.inject_var = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_blocks)
        )
        self.inject_sample = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_blocks)
        )
        for lin in (*self.inject_var, *self.inject_sample):
            nn.init.zeros_(lin.weight)

    def embed(self, x: Tensor, flag: Tensor) -> Optional[Tensor]:
        """Input-level flag embedding [B, Nc, D, d], or None in mode none."""
        if self.mode == "none":
            return None
        return flag[..., None].to(x.dtype) * self.flag_embed(x)

    def code(self, x: Tensor, flag: Tensor) -> Optional[Tensor]:
        """Flag code [B, Nc, D, d] (zero off intervened cells), or None."""
        if self.mode == "none":
            return None
        return flag[..., None].to(x.dtype) * self.flag_code(x)

    def inject(
        self, code: Optional[Tensor], block: int, where: str
    ) -> Optional[Tensor]:
        """Addition to the context tokens before step 1 (where="var") or step 2
        (where="sample") of `block`; None in mode none."""
        if code is None:
            return None
        maps = self.inject_var if where == "var" else self.inject_sample
        return maps[block](code)
