"""
File for random utility functions.
"""

import random
from datetime import datetime

import numpy as np
import torch


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
