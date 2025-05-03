from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch as th


@dataclass
class BaseModelConfig:
    d_model: int
    emb_depth: int
    dim_feedforward: int
    nhead: int
    dropout: float
    num_layers_encoder: int
    num_nodes: int
    sample_attn_mode: str
    linear_attention: bool
    mean_loss_across_samples: bool
    device: str
    dtype: torch.dtype


@dataclass
class CausalInfModelConfig(BaseModelConfig):
    num_mixture_components: Optional[int] = 1


@dataclass
class LocalLatentConfig(BaseModelConfig):
    decoder_depth: int
    num_z_samples_train: int
    num_z_samples_eval: int


@dataclass
class DataConfig:
    """Configuration related to datasets and dataloaders."""

    batch_size: int
    num_workers: int
    cntxt_split: List[float]
    sample_size: int
    train_dtype: str
    eval_dtype: str
    pin_memory: bool = True
    normalise: bool = True


@dataclass
class OptimizerConfig:
    """Configuration related to the optimizer and scheduler."""

    optimizer: th.optim.Optimizer
    learning_rate: float
    lr_warmup_ratio: float
    scheduler: Optional[th.optim.lr_scheduler._LRScheduler] = None


@dataclass
class TrainingConfig:
    """Configuration related to the training process."""

    epochs: int
    gradient_clip_val: Optional[float] = 1.0
    device: str = "cuda" if th.cuda.is_available() else "cpu"


@dataclass
class LoggingConfig:
    """Configuration related to logging and saving."""

    save_dir: Path
    use_wandb: bool = True
    log_step: int = 500
    save_checkpoint_every_n_steps: int = 1000
    plot_validation_samples: bool = True
    num_validation_plots: int = 10
