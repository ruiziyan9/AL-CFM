"""
File for random utility functions.
"""

from datetime import datetime
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
