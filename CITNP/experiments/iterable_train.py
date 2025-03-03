"""
File to train the model.
"""
import argparse

from torch.utils.data import DataLoader

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator
from CITNP.models.causalinferencemodel import CausalInferenceModel
from CITNP.utils.default_args import return_default_args


def main(args: argparse.Namespace):

    DATA_ARGS = {
        "sample_size": args.sample_size,
        "num_variables": args.num_variables,
        "function_generator": args.function_generator,
        "graph_type":  args.graph_type,
        "graph_degrees": args.graph_degrees,
    }

    dataset = InterventionDatasetGenerator(
        **DATA_ARGS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
    )

    MODEL_ARGS = {
        "d_model": args.d_model,
        "emb_depth": args.emb_depth,
        "dim_feedforward": args.dim_feedforward,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "num_layers_encoder": args.num_layers_encoder,
        "num_layers_decoder": args.num_layers_decoder,
        "num_nodes": args.num_nodes,
        "device": args.device,
        "dtype": args.dtype,
    }

    model = CausalInferenceModel(
        **MODEL_ARGS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_args = return_default_args()
    for key, value in default_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    main(args)
