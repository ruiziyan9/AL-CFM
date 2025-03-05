"""
Default arguments for the CITNP package.
"""
import torch


def return_default_args():
    """
    Returns the default arguments for the CITNP package.
    """
    return {
        "seed": 0,
        "d_model": 512,
        "emb_depth": 2,
        "dim_feedforward": 2048,
        "nhead": 8,
        "num_layers_encoder": 16,
        "device": "cuda",
        "dtype": torch.bfloat16,
        "batch_size": 32,
        "sample_size": 500,
        "num_variables": 2,
        "function_generator": "gp",
        "graph_type": ["ER"],
        "graph_degrees": [1],
        "iterations_per_epoch": 10000,
        "learning_rate": 0.0001,
    }