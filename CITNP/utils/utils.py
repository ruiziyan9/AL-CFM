"""
File for random utility functions.
"""

import copy
import random
import re
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from CITNP.utils.configs import (
    DataConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
)


def _get_clones(
    modules: Union[nn.Module, partial, Tuple[Union[nn.Module, partial], ...]],
    repeats: int,
) -> nn.ModuleList:
    """
    Repeat and deepcopy one or more modules in a fixed order.

    Args:
        modules: A single nn.Module or a tuple of modules.
        repeats: Number of times to repeat the pattern.

    Returns:
        nn.ModuleList of length `repeats * len(modules)`
    """
    if isinstance(modules, (nn.Module, partial)):
        modules = (modules,)

    cloned = []
    for i in range(repeats * len(modules)):
        base = modules[i % len(modules)]
        if isinstance(base, partial):
            cloned.append(base())  # instantiate
        else:
            cloned.append(copy.deepcopy(base))
    return nn.ModuleList(cloned)


def reshape_for_sample_attention(src, B, S, N, D):
    return src.permute(0, 2, 1, 3).reshape(B * N, S, D)


def reshape_back_sample_attention(src, B, S, N, D):
    return src.reshape(B, N, S, D).permute(0, 2, 1, 3)


def reshape_for_node_attention(src, B, S, N, D):
    return src.reshape(B * S, N, D)


def reshape_back_node_attention(src, B, S, N, D):
    return src.reshape(B, S, N, D)


def send_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [send_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: send_to_device(v, device) for k, v in data.items()}
    else:
        return data


def set_seed(seed: int):
    """
    Set the seed for generating random numbers to ensure reproducibility.

    This function sets the seed for Python's random module, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    # Python built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _generate_experiment_name() -> str:
    adjectives = [
        "swift",
        "fierce",
        "gentle",
        "clever",
        "brave",
        "cunning",
        "loyal",
        "playful",
        "quiet",
        "bold",
        "majestic",
        "nimble",
        "mighty",
        "elegant",
        "sly",
        "energetic",
        "daring",
        "graceful",
        "wild",
        "fearless",
    ]

    animals = [
        "tiger",
        "eagle",
        "wolf",
        "panther",
        "falcon",
        "fox",
        "bear",
        "lion",
        "cheetah",
        "hawk",
        "otter",
        "deer",
        "jaguar",
        "leopard",
        "owl",
        "raven",
        "buffalo",
        "cougar",
        "lynx",
        "badger",
        "doggo",
        "squirrel",
        "piglet",
    ]

    adjective = random.choice(adjectives)
    animal = random.choice(animals)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    return f"{adjective}-{animal}-{timestamp}"


def set_all_except_batchindex_to_zero(
    data: torch.Tensor,
    batch_index: torch.Tensor,
):
    """
    Sets all target data to zero except for the target data corresponding
    to the batch index.

    Args:
    -----
    - data (torch.Tensor): Tensor of shape
        (batch_size, num_target, num_nodes, 1) containing target values.
    - batch_index (torch.Tensor): Tensor of shape (batch_size,) with
            the index of the intervention node for each sample.

    Returns:
    --------
        torch.Tensor: Modified data with non-intervention targets set
        to zero.
    """
    device = data.device
    dtype = data.dtype
    data = data.clone().to(device)
    batch_size, _, num_nodes, _ = data.shape

    # Create a tensor that lists node indices for broadcasting.
    # Shape: (1, 1, num_nodes, 1)
    node_range = torch.arange(num_nodes, device=data.device).view(1, 1, num_nodes, 1)

    # Reshape intervention_index to (batch_size, 1, 1, 1) for broadcasting.
    intervention_index = batch_index.view(batch_size, 1, 1, 1)

    # Create a mask: for each sample, only the intervention node gets a 1; the rest are 0.
    mask = (
        node_range == intervention_index
    ).float()  # Shape: (batch_size, 1, num_nodes, 1)

    # Broadcast the mask along the num_target dimension and apply it.
    data = data * mask
    data = data.to(dtype)
    return data


def append_treatment_outcome_index_to_data(
    data: torch.Tensor,
    outcome_index: torch.Tensor,
    treatment_index: torch.Tensor,
):
    """
    Appends the treatment and outcome index to the data.
    The outcome is encoded by the value 1 and the treatment is encoded by
    the value -1. The rest of the nodes are encoded by 0.

    Args:
    -----
    - data (torch.Tensor): Context data tensor with shape
        (batch_size, num_samples, num_nodes, 1)
    - outcome_index (torch.Tensor): Outcome index tensor with shape
        (batch_size)
    - treatment_index (torch.Tensor): Treatment index tensor with shape
        (batch_size)
    """
    dtype = data.dtype
    batch_size = outcome_index.size(0)
    num_nodes = data.size(2)

    # Create a tensor of zeros with shape (batch_size, num_nodes)
    encoding = torch.zeros(batch_size, num_nodes, device=outcome_index.device)

    # Set outcome index to 1 for each batch sample
    encoding[torch.arange(batch_size), outcome_index] = 1

    # Set treatment index to -1 for each batch sample
    encoding[torch.arange(batch_size), treatment_index] = -1

    # Reshape to (batch_size, 1, num_nodes, 1) for concatenation
    encoding = encoding.unsqueeze(1).unsqueeze(-1)

    # Concatenate the single encoding tensor along the channel dimension
    encoded_data = torch.cat([data, encoding], dim=1)

    # Convert back to the original dtype if necessary
    encoded_data = encoded_data.to(dtype)

    return encoded_data


def gather_by_node(tensor, node_index):
    batch_idx = torch.arange(tensor.size(0), device=tensor.device)
    return tensor[batch_idx, :, node_index, :]


def load_latest_checkpoint(model, model_save_dir):
    model_save_dir = Path(model_save_dir)
    checkpoint_files = list(model_save_dir.glob("model_*.pt"))

    if not checkpoint_files:
        print("No checkpoint found.")
        raise FileNotFoundError(f"No checkpoint found in {model_save_dir}.")
        return model, None

    # Extract epoch number from filenames like 'model_12.pt'
    def extract_epoch(file_path):
        match = re.search(r"model_(\d+)\.pt", file_path.name)
        return int(match.group(1)) if match else -1

    latest_ckpt = max(checkpoint_files, key=extract_epoch)
    print(f"Loading checkpoint: {latest_ckpt.name}")

    model.load_state_dict(torch.load(latest_ckpt))
    return model, latest_ckpt


def _has_path(adj: torch.Tensor, start: int, end: int) -> bool:
    """
    Depth‐first search on a single adjacency matrix `adj` (shape [N,N]),
    checking for a directed path start → end.
    """
    N = adj.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            return True
        if not visited[node]:
            visited[node] = True
            # find children of `node`
            children = (adj[node] != 0).nonzero(as_tuple=False).view(-1)
            for c in children.tolist():
                if not visited[c]:
                    stack.append(c)
    return False


def get_causal_direction(
    graphs: torch.Tensor,  # [B, N, N] adjacency matrices
    intvn_indices: torch.Tensor,  # [B] intervention node indices
    outcome_indices: torch.Tensor,  # [B] outcome node indices
) -> List[str]:
    """
    Returns a list of length B with values in {"upstream","downstream","independent"}.
    """
    B = graphs.size(0)
    dirs = []
    for b in range(B):
        adj = graphs[b]
        i = int(intvn_indices[b].item())
        o = int(outcome_indices[b].item())

        if _has_path(adj, i, o):
            dirs.append("upstream")
        elif _has_path(adj, o, i):
            dirs.append("downstream")
        else:
            dirs.append("independent")
    return dirs


def convert_trainerdict_to_config(
    trainer_dict: dict,
) -> dict:
    """
    Converts a trainer dictionary to a configuration dictionary.

    Args:
        trainer_dict (dict): The trainer dictionary to convert.

    Returns:
        dict: The converted configuration dictionary.
    """
    dataconfig = DataConfig(
        batch_size=trainer_dict["batch_size"],
        num_workers=trainer_dict["num_workers"],
        cntxt_split=trainer_dict["cntxt_split"],
        sample_size=trainer_dict["sample_size"],
        train_dtype=trainer_dict["train_dtype"],
        eval_dtype=trainer_dict["eval_dtype"],
        pin_memory=trainer_dict["pin_memory"],
        normalise=trainer_dict["normalise"],
    )
    optimconfig = OptimizerConfig(
        optimizer=trainer_dict["optimizer"],
        scheduler=None,
        learning_rate=trainer_dict["learning_rate"],
        lr_warmup_ratio=trainer_dict["lr_warmup_ratio"],
    )
    trainingconfig = TrainingConfig(
        epochs=trainer_dict["epochs"],
        gradient_clip_val=trainer_dict["gradient_clip_val"],
        device=trainer_dict["device"],
    )
    loggingconfig = LoggingConfig(
        save_dir=trainer_dict["save_dir"],
        use_wandb=trainer_dict["use_wandb"],
        log_step=trainer_dict["log_step"],
        save_checkpoint_every_n_steps=trainer_dict["save_checkpoint_every_n_steps"],
        plot_validation_samples=trainer_dict["plot_validation_samples"],
        num_validation_plots=trainer_dict["num_validation_plots"],
    )
    return dataconfig, optimconfig, trainingconfig, loggingconfig
