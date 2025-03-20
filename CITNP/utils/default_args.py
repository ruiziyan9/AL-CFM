"""
Default arguments for the CITNP package.
"""

import argparse

import torch


def parse_args():
    """
    Parses command-line arguments for the CITNP package, using defaults that
    match return_default_args() from your script.
    """
    parser = argparse.ArgumentParser(
        description="Default arguments for the CITNP package."
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random number generation."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode.",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension."
    )
    parser.add_argument(
        "--emb_depth", type=int, default=2, help="Embedding depth."
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Feedforward network dimension.",
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads."
    )
    parser.add_argument(
        "--num_layers_encoder",
        type=int,
        default=8,
        help="Number of layers in the encoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Computation device to use (e.g., "cuda" or "cpu").',
    )

    # For dtype, pass a string such as "bfloat16", then convert it to torch.dtype.
    # You could customize str_to_dtype if you want to restrict or handle other dtypes.
    def str_to_dtype(s):
        return getattr(torch, s, torch.bfloat16)

    parser.add_argument(
        "--dtype",
        type=str_to_dtype,
        default="bfloat16",
        help='Torch data type (e.g., "bfloat16" or "float32").',
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Number of samples."
    )
    parser.add_argument(
        "--num_variables", type=int, default=2, help="Number of variables."
    )
    parser.add_argument(
        "--function_generator",
        type=str,
        default="sinusoid",
        help="Type of function to generate.",
    )
    parser.add_argument(
        "--graph_type", nargs="+", default=["ER"], help="Graph type(s)."
    )
    parser.add_argument(
        "--graph_degrees",
        nargs="+",
        type=int,
        default=[1],
        help="Degree(s) for the graph.",
    )
    parser.add_argument(
        "--iterations_per_epoch",
        type=int,
        default=10000,
        help="Number of iterations per epoch.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run."
    )

    # Training args
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs."
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Dont use Weights & Biases.",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./experiments",
        help="Path where to store the results.",
    )
    parser.add_argument("--log_step", type=int, default=500, help="Log step.")

    parser.add_argument(
        "--save_checkpoint_every",
        type=int,
        default=1,
        help="Number of epochs after which to save checkpoint.",
    )

    return parser.parse_args()


def overwrite_debug_args(args):
    """
    Overwrites the arguments when running in debug mode.
    """
    args.sample_size = 100
    args.iterations_per_epoch = 100
    args.epochs = 1
    args.d_model = 64
    args.dim_feedforward = 128
    args.num_layers_encoder = 2
    args.lr_warmup_ratio = 0.0
    args.no_wandb = True
    args.run_name = "debug"
    args.nhead = 4
    return args
