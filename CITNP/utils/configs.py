from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CausalInfModelConfig:
    d_model: int
    emb_depth: int
    dim_feedforward: int
    nhead: int
    dropout: float
    num_layers_encoder: int
    num_nodes: int
    mean_loss_across_samples: bool
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


@dataclass
class LocalLatentConfig:
    d_model: int
    emb_depth: int
    dim_feedforward: int
    nhead: int
    dropout: float
    num_layers_encoder: int
    num_nodes: int
    decoder_depth: int
    num_z_samples_train: int
    num_z_samples_eval: int
    mean_loss_across_samples: bool
    use_vi: bool = True
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
