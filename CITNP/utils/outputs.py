"""
Classes containing structured outputs for the model classes.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    pred_mean: torch.Tensor
    pred_std: torch.Tensor
