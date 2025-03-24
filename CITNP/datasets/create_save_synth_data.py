"""
File to create and save synthetic training data.
"""

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
        save_name = f"{function_generator}_{graph_type}_{num_variables}_{sample_size}"
        save_folder = save_dir / save_name
        save_folder.mkdir(exist_ok=True, parents=True)

        save_data(
            save_folder / f"data_idx{i}",
            all_data,
        )
    with open(save_folder / "generator_args.json", "w") as f:
        json.dump(GENERATOR_ARGS, f)


if __name__ == "__main__":
    save_dir = Path("synth_training_data")
    save_dir.mkdir(parents=True, exist_ok=True)

    data_start = 0
    data_end = 10

    main(save_dir=save_dir, data_start=data_start, data_end=data_end)
