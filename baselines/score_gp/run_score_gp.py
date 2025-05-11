"""
Baseline where we run SCORE causal discovery and then fit GPs onto the single
graph.
"""
import argparse
import sys
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pandas as pd
from dodiscover import make_context
from dodiscover.toporder.nogam import NoGAM

# Put arco in path
arcopath = "/vol/bitbucket/ad6013/Research/bci-arco-gp"
sys.path.append(arcopath)
arcodibsgp_path = "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp"
sys.path.append(arcodibsgp_path)

from run_arcogp import get_intvn_index, load_data, load_data_into_environment
from src.environments.environment import Environment
from src.environments.experiment import Experiment
from src.mechanism_models.mechanisms import get_mechanism_key
from src.mechanism_models.shared_data_gp_model import \
    SharedDataGaussianProcessModel
from src.utils.causal_orders import co_from_graph


def discover_causal_graph(obs_data: pd.DataFrame) -> nx.DiGraph:
    context = make_context().variables(data=obs_data).build()
    nogam = NoGAM()
    nogam.learn_graph(obs_data, context=context)
    graph = nogam.graph_
    return graph


def train_causal_model(
    env: Environment,
    graph: nx.DiGraph,
):
    parents_dict = {}
    for node in graph.nodes():
        parents_dict[node] = list(graph.predecessors(node))

    mechanism_keys = set()
    mechanisms = []
    for node in env.node_labels:
        mechanisms.extend([get_mechanism_key(node, parents_dict[node])])
    mechanism_keys.update(set(mechanisms))
    mechanism_keys = list(mechanism_keys)

    experiments = env.observational_train_data

    mechanism_model = SharedDataGaussianProcessModel(env.node_labels)

    mechanism_model.clear_prior_mll_cache()
    mechanism_model.clear_posterior_mll_cache()
    mechanism_model.clear_rmse_cache()

    mechanism_model.init_mechanisms_from_keys(mechanism_keys, 1)

    mechanism_model.discard_gps()
    mechanism_model.update_gp_hyperparameters(experiments, mechanism_keys)


def main(data_path: Path, num_datasets: int, sample_size: int, data_start: int, data_end: int):

    for i in range(data_start, data_end):
        graph, obs_train, int_test = load_data(data_path, i)
        intvn_index = get_intvn_index(int_test)
        env = load_data_into_environment(
            obs_train=obs_train,
            int_test=int_test,
            graph=graph,
            sample_size=sample_size,
            intvn_index=intvn_index,
        )

        causal_graph = discover_causal_graph(
            obs_data=obs_train[:sample_size],
        )
        train_causal_model(
            env=env,
            graph=causal_graph,
        )
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    data_folder = Path(
        "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp/datasets"
    )
    data_name = "30var_ER4_neuralgplvm_1000"
    data_path = data_folder / data_name

    num_datasets = 100
    sample_size = 500

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_start",
        type=int,
        default=0,
        help="data_start",
    )
    parser.add_argument(
        "--data_end",
        type=int,
        default=100,
        help="data_end",
    )
    args = parser.parse_args()

    main(
        data_path=data_path,
        num_datasets=num_datasets,
        sample_size=sample_size,
        data_start=args.data_start,
        data_end=args.data_end
    )