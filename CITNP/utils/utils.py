"""
File for random utility functions.
"""
import random

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