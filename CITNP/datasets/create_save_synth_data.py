"""
File to create and save synthetic training data.
"""

import argparse
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import trange

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator


def save_data(save_path: Path, data: dict):
    with h5py.File(f"{save_path}.hdf5", "w") as f:
        for key, value in data.items():
            if value is None:
                continue
            f.create_dataset(key, data=value)


def generate_and_save_data(
    generator,
    generator_args,
    data_start: int,
    data_end: int,
    save_dir: Path,
    folder: str,
    name_prefix: str = "",
):
    for i in trange(data_start, data_end):
        result = next(generator.generate_next_dataset())
        (
            obs_data_array,
            int_data_array,
            causal_graphs_array,
            intvn_indices,
            variable_counts,
            masks,
        ) = result[:6]

        # If functions are included in the dataset, extract them
        functions = result[6] if len(result) > 6 else None

        all_data = {
            "obs_data": obs_data_array,
            "int_data": int_data_array,
            "causal_graphs": causal_graphs_array,
            "intvn_indices": intvn_indices,
            "variable_counts": variable_counts,
            "masks": masks,
        }

        # Add functions if available
        if functions is not None:
            all_data["functions"] = functions

        fn = generator_args["function_generator"]
        nv = generator_args["num_variables"]
        ss = generator_args["sample_size"]
        save_name = f"{name_prefix}_{fn}_{ss}"
        save_folder = save_dir / save_name / folder
        save_folder.mkdir(exist_ok=True, parents=True)

        save_data(save_folder / f"data_idx{nv}", all_data)

    with open(save_folder / "generator_args.json", "w") as f:
        json.dump(generator_args, f)


def main(
    save_dir: Path,
    data_start: int,
    data_end: int,
    split: str,
    name_prefix: str = "",
    sample_size: int = 1000,
    num_variables: List[int] | int = 50,
    function_generator: str = "neuralgplvm",
    graph_type: List[str] = ["ER", "SF"],
    graph_degrees: List[int] | dict = [50, 100, 150, 200, 250],
):
    batch_sizes = {"train": 8 * 8000, "val": 32, "test": 128}

    if split not in batch_sizes:
        raise ValueError(
            f"Invalid split '{split}'. Choose from 'train', 'val', 'test'."
        )

    GENERATOR_ARGS = {
        "sample_size": sample_size,
        "num_variables": num_variables,
        "function_generator": function_generator,
        "graph_type": graph_type,
        "graph_degrees": graph_degrees,
        "iterations_per_epoch": 1,
        "batch_size": batch_sizes[split],
        "normalise": True,
        "return_functions": False,
        "show_progress": True,
    }

    generator = InterventionDatasetGenerator(**GENERATOR_ARGS)  # type: ignore

    generate_and_save_data(
        generator,
        GENERATOR_ARGS,
        data_start=data_start,
        data_end=data_end,
        save_dir=save_dir,
        folder=split,
        name_prefix=name_prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_start", type=int, default=0, help="Index to start saving data."
    )
    parser.add_argument(
        "--data_end", type=int, default=1, help="Index to end saving data."
    )
    parser.add_argument("--var_start", type=int, default=5)
    parser.add_argument("--var_end", type=int, default=51)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to create.",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="",
        help="Optional prefix to prepend to save folder name.",
    )
    args = parser.parse_args()

    save_dir = Path("synth_training_data")
    save_dir.mkdir(parents=True, exist_ok=True)

    few_nodes_graph_degrees = {
        5: np.arange(5, 10, dtype=int).tolist(),
        6: np.arange(6, 15, dtype=int).tolist(),
        7: np.arange(7, 21, dtype=int).tolist(),
        8: np.arange(8, 28, dtype=int).tolist(),
        9: np.arange(9, 36, dtype=int).tolist(),
        10: np.arange(10, 45, dtype=int).tolist(),
        11: np.arange(11, 55, dtype=int).tolist(),
        12: np.arange(12, 66, dtype=int).tolist(),
    }
    large_graph_degrees = {
        k: np.arange(k, 6 * k, dtype=int).tolist() for k in range(13, 51)
    }
    graph_degrees = {**few_nodes_graph_degrees, **large_graph_degrees}

    all_num_var = np.arange(args.var_start, args.var_end, dtype=int).tolist()

    for num_var in all_num_var:
        main(
            save_dir=save_dir,
            data_start=args.data_start,
            data_end=args.data_end,
            split=args.split,
            name_prefix=args.name_prefix,
            num_variables=num_var,
            graph_degrees=graph_degrees,
        )
