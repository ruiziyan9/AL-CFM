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
        "--work_dir",
        type=str,
        # default="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP",
        default="/scratches/cartwright/cdd43/CausalInferenceNeuralProcess/CITNP",
        help="Path to the working directory.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        # default="anticausal_simplegplvm_['ER']_2_1000",
        default="simplegp_['ER']_3_1000",
        help="Name of the data file to use.",
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
        "--model_type", choices=["cnp", "locallatent"], default="cnp"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Model dimension."
    )
    parser.add_argument(
        "--emb_depth", type=int, default=2, help="Embedding depth."
    )
    parser.add_argument(
        "--decoder_depth", type=int, default=0, help="Decoder depth."
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
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
        "--sample_attn_mode",
        type=str,
        default="MHSA",
        help="Type of sample attention: MHSA or MHCA.",
    )
    parser.add_argument(
        "--linear_attention",
        action="store_true",
        default=False,
        help="Whether to use linear attention.",
    )
    parser.add_argument(
        "--mean_loss_across_samples",
        action="store_true",
        default=True,
        help="Whether to average the loss across samples.",
    )
    # MoG args
    parser.add_argument(
        "--num_mixture_components",
        type=int,
        default=1,
        help="Number of components in MoG.",
    )
    # Local VI args
    parser.add_argument(
        "--num_z_samples_train",
        type=int,
        default=16,
        help="Number of z samples for training.",
    )
    parser.add_argument(
        "--num_z_samples_eval",
        type=int,
        default=16,
        help="Number of z samples for evaluation.",
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

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Number of samples."
    )
    parser.add_argument(
        "--num_variables", type=int, default=2, help="Number of variables."
    )
    parser.add_argument(
        "--function_generator",
        type=str,
        default="simplegp",
        help="Type of function to generate.",
    )
    parser.add_argument(
        "--graph_type", nargs="+", default=["ER"], help="Graph type(s)."
    )
    parser.add_argument(
        "--graph_degrees",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Degree(s) for the graph.",
    )
    parser.add_argument(
        "--iterations_per_epoch",
        type=int,
        default=10000,
        help="Number of iterations per epoch.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Learning rate."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run."
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Dont use Weights & Biases.",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.02,
        help="Warmup ratio for learning rate.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./experiments",
        help="Path where to store the results.",
    )
    parser.add_argument("--log_step", type=int, default=200, help="Log step.")

    parser.add_argument(
        "--save_checkpoint_every",
        type=int,
        default=1,
        help="Number of epochs after which to save checkpoint.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading.",
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
