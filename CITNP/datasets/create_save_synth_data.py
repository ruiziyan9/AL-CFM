"""
File to create and save synthetic training data.
"""

import argparse
import json
from pathlib import Path
from typing import List

import h5py
from tqdm import trange

from CITNP.datasets.dataset_generator import InterventionDatasetGenerator


def save_data(save_path: Path, data: dict):
    with h5py.File(f"{save_path}.hdf5", "w") as f:
        for key, value in data.items():
            if value is None:
                continue
            f.create_dataset(key, data=value)


def generate_save_data(
    generator,
    generator_args,
    data_start: int,
    data_end: int,
    save_dir: Path,
    folder="train",
):
    for i in trange(data_start, data_end):
        (
            obs_data_array,
            int_data_array,
            causal_graphs_array,
            intvn_indices,
            variable_counts,
            masks,
        ) = next(generator.generate_next_dataset())
        all_data = {
            "obs_data": obs_data_array,
            "int_data": int_data_array,
            "causal_graphs": causal_graphs_array,
            "intvn_indices": intvn_indices,
            "variable_counts": variable_counts,
            "masks": masks,
        }
        fn = generator_args["function_generator"]
        gt = generator_args["graph_type"]
        nv = generator_args["num_variables"]
        ss = generator_args["sample_size"]
        save_name = f"{fn}_{gt}_{nv}_{ss}"
        save_folder = save_dir / save_name / folder
        save_folder.mkdir(exist_ok=True, parents=True)

        save_data(
            save_folder / f"data_idx{i}",
            all_data,
        )
    with open(save_folder / "generator_args.json", "w") as f:
        json.dump(generator_args, f)


def main(
    save_dir: Path,
    data_start: int,
    data_end: int,
    sample_size: int = 1000,
    num_variables: List[int] = 2,
    function_generator: str = "simplegplvm",
    graph_type: List[str] = ["ER"],
    batch_size: int = 50000,
    graph_degrees: List[int] = [1],
):

    GENERATOR_ARGS = {
        "sample_size": sample_size,
        "num_variables": num_variables,
        "function_generator": function_generator,
        "graph_type": graph_type,
        "graph_degrees": graph_degrees,
        "iterations_per_epoch": 1,
        "batch_size": batch_size,
    }

    generator = InterventionDatasetGenerator(**GENERATOR_ARGS)

    generate_save_data(
        generator,
        GENERATOR_ARGS,
        data_start=data_start,
        data_end=data_end,
        save_dir=save_dir,
        folder="train",
    )

    GENERATOR_ARGS["batch_size"] = 100
    generator = InterventionDatasetGenerator(**GENERATOR_ARGS)
    generate_save_data(
        generator,
        GENERATOR_ARGS,
        data_start=0,
        data_end=1,
        save_dir=save_dir,
        folder="val",
    )

    GENERATOR_ARGS["batch_size"] = 25
    generator = InterventionDatasetGenerator(**GENERATOR_ARGS)
    generate_save_data(
        generator,
        GENERATOR_ARGS,
        data_start=0,
        data_end=1,
        save_dir=save_dir,
        folder="test",
    )


if __name__ == "__main__":
    save_dir = Path("synth_training_data")
    save_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_start",
        type=int,
        default=0,
        help="Index to start saving data.",
    )
    parser.add_argument(
        "--data_end",
        type=int,
        default=10,
        help="Index to end saving data.",
    )
    args = parser.parse_args()

    main(save_dir=save_dir, data_start=args.data_start, data_end=args.data_end)
