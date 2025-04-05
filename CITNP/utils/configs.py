from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Config:
    d_model: int
    emb_depth: int
    dim_feedforward: int
    nhead: int
    dropout: float
    num_layers_encoder: int
    num_nodes: int
    mean_loss_across_samples: bool
    device: str
    dtype: torch.dtype


@dataclass
class CausalInfModelConfig(Config):
    num_mixture_components: Optional[int] = 1


@dataclass
class LocalLatentConfig(Config):
    decoder_depth: int
    num_z_samples_train: int
    num_z_samples_eval: int
