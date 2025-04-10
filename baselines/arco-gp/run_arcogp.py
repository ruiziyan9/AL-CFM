"""
We use the following code to run the ARCO-GP algorithm on a given dataset.

Metric:
- log p(y' | do(x'), D) for each dataset across all the intervention samples
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

# Put arco in path
arcopath = "/vol/bitbucket/ad6013/Research/bci-arco-gp"
sys.path.append(arcopath)

from src.abci_arco_gp import ABCIArCOGP as ABCI
from src.config import ABCIArCOGPConfig
from src.environments.environment import Environment
from src.utils.graphs import adj_mat_to_graph, get_parents, graph_from_csv


def get_config():
    cfg = ABCIArCOGPConfig()
    cfg.policy = "static-obs-dataset"
    cfg.max_ps_size = 2
    cfg.num_arco_steps = 200
    cfg.num_mc_cos = 50
    cfg.num_mc_graphs = 5
    cfg.num_samples_per_graph = 5
    cfg.compute_distributional_stats = False
    return cfg


def load_data(data_path: Path, dataset_index: int):
    data_name = data_path.name
    graph_path = data_path / f"{data_name}_adjmat_{dataset_index}.csv"
    graph = graph_from_csv(graph_path)

    obs_data_path = data_path / f"{data_name}_obs_data_{dataset_index}.csv"
    obs_train = pd.read_csv(obs_data_path)

    int_data_path = data_path / f"{data_name}_int_data_{dataset_index}.csv"
    int_test = pd.read_csv(int_data_path)
    return graph, obs_train, int_test


def get_intvn_index(int_test: pd.DataFrame):
    # Get the index of the intervention variable
    for i in range(len(int_test.columns)):
        if int_test.iloc[0, i] == 0:
            intvn_index = i
            break
    return intvn_index


def create_test_experiments(int_test: pd.DataFrame, sample_size: int, intvn_index: int):
    # Create a list of experiments
    all_intvn_test_data = []
    for i in range(sample_size):
        test_data_frame = pd.DataFrame(
            columns=int_test.columns, data=int_test.iloc[0:1].copy()
        )
        current_intvn_test_data = int_test.iloc[i + 1 : i + 2].copy()

        # change the intervention value to the current one
        test_data_frame.iloc[0, intvn_index] = current_intvn_test_data.iloc[
            0, intvn_index
        ]

        final_intvn_test_data = pd.concat(
            [test_data_frame, current_intvn_test_data], axis=0
        )

        all_intvn_test_data.append(final_intvn_test_data)
    return all_intvn_test_data


def load_data_into_environment(
    obs_train: pd.DataFrame,
    int_test: pd.DataFrame,
    graph: pd.DataFrame,
    sample_size: int,
    intvn_index: int,
):
    obs_train = obs_train.loc[0 : sample_size - 1]
    int_test = int_test.loc[0:sample_size]

    all_intvn_test_data = create_test_experiments(
        int_test=int_test,
        sample_size=sample_size,
        intvn_index=intvn_index,
    )

    loaded_env = Environment.load_static_dataset(
        graph=graph,
        obs_train_data=obs_train,
        intr_test_data=all_intvn_test_data,
        normalise=True,
    )
    return loaded_env


def calculate_nlpd(model: ABCI, intvn_index: int):
    # record interventional test LLs
    # Get the node to test
    node = "X0" if intvn_index != 0 else "X1"

    def test_ll(adj_mat):
        graph = adj_mat_to_graph(adj_mat, model.mechanism_model.node_labels)
        model.mechanism_model.init_topological_order(graph, model.sample_time)
        parents = get_parents(node, graph)
        return model.mechanism_model.node_mll(
            model.env.interventional_test_data,
            prior_mode=False,
            use_cache=True,
            mode="independent_samples",
            node=node,
            parents=parents,
            reduce=False,
        )

    model.mechanism_model.clear_posterior_mll_cache()
    with torch.no_grad():
        node_ll = model.graph_posterior_expectation_mc(test_ll)
        # Calculate the log likelihood
    return -node_ll.sum().item()


def main(data_path: Path, num_datasets: int, sample_size: int):

    config = get_config()

    all_nlpd = []
    for i in trange(num_datasets):
        # Load data
        graph, obs_train, int_test = load_data(data_path, i)
        intvn_index = get_intvn_index(int_test)
        env = load_data_into_environment(
            obs_train=obs_train,
            int_test=int_test,
            graph=graph,
            sample_size=sample_size,
            intvn_index=intvn_index,
        )

        # Set up the model
        model = ABCI(
            env=env,
            cfg=config,
        )

        # Run the model
        model.run()

        nlpd = calculate_nlpd(
            model=model,
            intvn_index=intvn_index,
        )
        all_nlpd.append(nlpd)
        print(f"Dataset {i}: NLL = {nlpd}")

    print(f"Average NLL: {sum(all_nlpd) / len(all_nlpd)}")

    # Save the results
    save_name = data_path.name + f"_nlpd.csv"
    save_folder = data_path.parent.parent / "results"
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / save_name
    df = pd.DataFrame(all_nlpd, columns=["NLL"])
    df.to_csv(save_path, index=False)


if __name__ == "__main__":

    data_folder = Path(
        "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-gp/datasets"
    )
    data_name = "3var_simplegp_ER_3_1000"
    data_path = data_folder / data_name

    num_datasets = 25
    sample_size = 500

    main(data_path=data_path, num_datasets=num_datasets, sample_size=sample_size)
