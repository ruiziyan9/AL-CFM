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
arcodibsgp_path = (
    "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp"
)
sys.path.append(arcodibsgp_path)

from run_arcogp import get_intvn_index, load_data, load_data_into_environment
from src.environments.environment import Environment
from src.environments.experiment import Experiment
from src.mechanism_models.mechanisms import get_mechanism_key
from src.mechanism_models.shared_data_gp_model import \
    SharedDataGaussianProcessModel
from src.utils.graphs import get_parents


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
    return mechanism_model


def calculate_nlpd(
    model: SharedDataGaussianProcessModel,
    intvn_index: int,
    graph: nx.DiGraph,
    env: Environment,
):
    outcome_node = "X0" if intvn_index != 0 else "X1"

    model.init_topological_order(graph, 2)
    parents = get_parents(outcome_node, graph)

    model.clear_posterior_mll_cache()
    outcome_mll = model.node_mll(
        env.interventional_test_data,
        prior_mode=False,
        use_cache=True,
        mode="independent_samples",
        node=outcome_node,
        parents=parents,
        reduce=False,
    )
    outcome_nlpd = -outcome_mll.sum().item()
    return outcome_nlpd


def main(
    data_path: Path, num_datasets: int, sample_size: int, data_start: int, data_end: int
):

    result_folder = (
        data_path.parent.parent.parent / "score_gp" / "results" / data_path.name
    )
    result_folder.mkdir(parents=True, exist_ok=True)

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
        causal_model = train_causal_model(
            env=env,
            graph=causal_graph,
        )
        nlpd = calculate_nlpd(
            model=causal_model,
            intvn_index=intvn_index,
            graph=causal_graph,
            env=env,
        )
        # Save the results
        save_name = data_path.name + f"_nogamgp_nlpd_data{i}.csv"
        save_path = result_folder / save_name
        df = pd.DataFrame([nlpd], columns=["NLL"])
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    data_folder = Path(
        "/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp/datasets"
    )
    data_name = "20var_ER4_neuralgplvm_1000"
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
        data_end=args.data_end,
    )
