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
        "d_model": 64,
        "emb_depth": 64,
        "dim_feedforward": 128,
        "nhead": 4,
        "dropout": 0.1,
        "num_layers_encoder": 2,
        "num_layers_decoder": 2,
        "num_nodes": 10,
        "device": "cuda",
        "dtype": torch.bfloat16,
        "sample_size": 500,
        "num_variables": 3,
        "function_generator": "gp",
        "graph_type": ["ER"],
        "graph_degrees": [3, 2, 1],
    }