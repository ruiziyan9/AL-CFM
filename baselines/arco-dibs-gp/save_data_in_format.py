"""
Save data in format that can be used by the ARCO-GP model.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_data(data_path: Path):
    with h5py.File(data_path, "r") as f:
        obs_data = f["obs_data"][:]
        int_data = f["int_data"][:]
        causal_graphs = f["causal_graphs"][:]
        intvn_indices = f["intvn_indices"][:]
        variable_counts = f["variable_counts"][:]

    data = {
        "obs_data": obs_data,
        "int_data": int_data,
        "causal_graphs": causal_graphs,
        "intvn_indices": intvn_indices,
        "variable_counts": variable_counts,
    }
    return data


def save_dataframe(
    df: pd.DataFrame,
    save_path: Path,
    suffix: str,
):
    """
    Save dataframe to csv file.
    """
    save_name = save_path.name + f"_{suffix}.csv"
    save_folder = save_path
    save_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_folder / save_name, index=False)


def create_header(num_vars: int):
    header = []
    for i in range(num_vars):
        if isinstance(i, int):
            header.append(f"X{i}")
        else:
            header.append(i.item())
    # header = [f"X{i.item()}" for i in range(num_vars)]
    return header


def format_obs_data(unformatted_obs_data: np.ndarray, header: list):
    """
    Creates a dataframe with header as column headers.
    """
    # Set header
    obs_data = pd.DataFrame(
        unformatted_obs_data,
        columns=header,
    )
    return obs_data


def format_int_data(unformatted_int_data: np.ndarray, header: list, intvn_index: int):
    """
    Creates a dataframe with header as column headers.
    """
    # Set header
    int_data = pd.DataFrame(
        unformatted_int_data,
        columns=header,
    )
    # Add row to top indicating intervention variable
    no_intvn_indicator = ["n/i"] * len(header)
    no_intvn_indicator[intvn_index] = 0.0
    row_df = pd.DataFrame([no_intvn_indicator], columns=header)
    int_data = pd.concat([row_df, int_data], ignore_index=True)
    return int_data


def main(
    data_path: Path,
    save_folder: Path,
):
    # Load data
    data = load_data(data_path)
    num_vars = data["variable_counts"].item() -1

    header = create_header(num_vars)

    num_datasets = data["obs_data"].shape[0]

    save_path = save_folder / data_path.parent.parent.name

    for dataset in range(num_datasets):

        # Causal graph
        unformatted_causal_graph = data["causal_graphs"][dataset][:num_vars, :num_vars]
        causal_graph = format_obs_data(
            unformatted_causal_graph,
            header,
        )
        save_dataframe(causal_graph, save_path, suffix=f"adjmat_{dataset}")

        # Obs data
        unformatted_obs_data = data["obs_data"][dataset]
        obs_data = format_obs_data(unformatted_obs_data, header)
        save_dataframe(obs_data, save_path, suffix=f"obs_data_{dataset}")

        # Int data
        unformatted_int_data = data["int_data"][dataset]
        intvn_index = data["intvn_indices"][dataset].item()
        int_data = format_int_data(unformatted_int_data, header, intvn_index)
        save_dataframe(int_data, save_path, suffix=f"int_data_{dataset}")


if __name__ == "__main__":

    data_folder = "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/datasets/synth_training_data/sachs/test"
    data_path = Path(data_folder) / "data_idx0.hdf5"

    save_folder = Path(
        "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp/datasets"
    )

    main(data_path=data_path, save_folder=save_folder)
