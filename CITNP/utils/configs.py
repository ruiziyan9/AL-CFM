from dataclasses import dataclass

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
    device: str
    dtype: torch.dtype


@dataclass
class LocalLatentConfig(CausalInfModelConfig):
    decoder_depth: int
    num_z_samples_train: int
    num_z_samples_eval: int
