"""
Classes containing structured outputs for the model classes.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    pred_mean: torch.Tensor
    pred_std: torch.Tensor
    weights: Optional[torch.Tensor] = None
