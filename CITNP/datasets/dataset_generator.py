"""
This file contains dataset generators.
"""
from functools import partial
from typing import List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset
from tqdm import trange

from CITNP.datasets.causal_graph_generator import generate_synthetic_dag
from CITNP.datasets.functions_generator import GPFunctionGenerator


class InterventionDatasetGenerator(IterableDataset):
    """Generate datasets using a function generator. This will generate
    observational data along with interventional datasets.

    Args:
    ----------
    function_generator : str
        The function generator to use. This can be one of the following:
        - "gp": Gaussian process function generator.

    num_variables : Union[int, List[int]]
        The number of variables in the dataset. If a list is provided,
        the number of variables will be randomly sampled from this list for each dataset.

    sample_size : int
        The maximum number of samples to generate.

    batch_size : int
        The number of datasets to generate.

    graph_type : List[str]
        The type of graph to generate. This can be one
        or a list of the following. If it is a list, then for each
        dataset a different graph type will be sampled.
        - "ER": Erdos-Renyi
        - "SF": Scale-free, Barabasi-Albert

    graph_degrees : List[int]
        The expected degrees of the graph. If it is a list, then for eac
        dataset the degree will be sampled from this list.
    """

    def __init__(
        self,
        function_generator: str,
        num_variables: Union[int, List[int]],
        sample_size: int,
        batch_size: int,
        graph_type: List[str],
        graph_degrees: List[int],
    ):
        self.function_generator = function_generator
        self.num_variables = num_variables
        self.sample_size = sample_size

        self.batch_size = batch_size
        self.graph_type = graph_type
        self.graph_degrees = graph_degrees

        self.max_num_variables = max(num_variables)

        if self.function_generator == "gp":
            self.data_generator = partial(GPFunctionGenerator)

    def permute_data(self, *args, permutation_indices: np.ndarray) -> list:
        """
        Permute the data to randomise the causal graph.

        Args:
        ----------
        permutation_indices : np.ndarray shape (num_variables,)

        args : np.ndarray shape (num_samples, num_variables)
            Data to permute.

        Returns:
        ----------
        permuted_data : list of np.ndarray shape (num_samples, num_variables)
        """
        permuted_data = [data[:, permutation_indices] for data in args]
        return permuted_data

    def permute_causal_graph(
        self, *args, permutation_indices: np.ndarray
    ) -> list:
        """
        Permutes the causal graph.

        Args:
        ----------
        permutation_indices : np.ndarray shape (num_variables,)

        args : np.ndarray shape (num_variables, num_variables)
            Causal graph to permute.

        Returns:
        ----------
        permuted_causal_graphs : list of np.ndarray shape (num_variables, num_variables)
        """
        permuted_causal_graphs = [
            dag[permutation_indices, :][:, permutation_indices] for dag in args
        ]
        return permuted_causal_graphs

    def sample_uniform_expected_degree(self) -> int:
        """
        Sample a degree from the list of expected degrees.

        Returns:
        ----------
        degree : int
        """
        degree = np.random.choice(self.graph_degrees)
        return degree

    def pad_data(self, data: np.ndarray, current_vars: int) -> np.ndarray:
        """
        Pad data to match the maximum number of variables.

        Args:
        ----------
        data : np.ndarray
            Data to pad
        current_vars : int
            Current number of variables

        Returns:
        ----------
        padded_data : np.ndarray
            Padded data
        """
        if current_vars == self.max_variables:
            return data

        # For 2D data: samples x variables
        if data.ndim == 2:
            pad_width = ((0, 0), (0, self.max_variables - current_vars))
        # For 3D data: variables x variables (adjacency matrix)
        elif data.ndim == 3:
            pad_width = ((0, 0), (0, self.max_variables - current_vars),
                         (0, self.max_variables - current_vars))

        return np.pad(data, pad_width, mode='constant', constant_values=0)

    def generate_next_dataset(
        self,
    ):
        """
        Generate the next dataset point.

        This will be done by sampling a causal graph, and then using the data
        generation functions to generate the data.

        This will generate the observational and interventional data.

        Args:
        ----------

        Yields:
        ----------
        obs_data_array : np.ndarray shape (batch_size, sample_size, num_variables)
            The observational data.
        int_data_array : np.ndarray shape (batch_size, sample_size, num_variables)
            The interventional data.
        causal_graphs_array : np.ndarray shape (batch_size, num_variables, num_variables)
            The causal graphs.
        intervened_causal_graphs_array : np.ndarray shape (batch_size, num_variables, num_variables)
            The intervened causal graphs.
        variable_counts : np.ndarray shape (batch_size,)
            The number of variables in each dataset.
        masks : np.ndarray shape (batch_size, max_variables)
            Binary masks indicating which variables are valid (1) vs padding (0)
        """
        # Lists to store data for each batch
        obs_data_array = np.zeros(
            (self.batch_size, self.sample_size, self.max_variables)
        )
        int_data_array = np.zeros(
            (self.batch_size, self.sample_size, self.max_variables)
        )
        causal_graphs_array = np.zeros(
            (self.batch_size, self.max_variables, self.max_variables)
        )
        intervened_causal_graphs_array = np.zeros(
            (self.batch_size, self.num_variables, self.max_variables)
        )
        variable_counts = np.zeros(self.batch_size)
        masks = np.zeros((self.batch_size, self.max_variables))


        # Each batch is a different dataset
        for b in trange(self.batch_size):
            # Sample number of variables
            if isinstance(self.num_variables, list):
                current_num_variables = np.random.choice(self.num_variables)
            else:
                current_num_variables = self.num_variables

            # Sample graph type and expected degree
            expected_node_degree = self.sample_uniform_expected_degree()
            curr_graph_type = np.random.choice(self.graph_type)
            # Generate the causal graph
            dag = generate_synthetic_dag(
                d=current_num_variables,
                s0=expected_node_degree,
                graph_type=curr_graph_type,
            )

            # Generate data
            currnet_data_generator = self.data_generator(
                number_of_variables=current_num_variables,
            )
            (
                obs_data,
                intvn_data,
                intvt_dag,
            ) = currnet_data_generator.generate_data(
                causal_graph=dag,
                sample_size=self.sample_size,
            )

            # Permute the data and graphs
            permutation_indices = np.random.permutation(current_num_variables)
            dag = self.permute_causal_graph(
                dag,
                permutation_indices=permutation_indices,
            )[0]

            intvt_dag = self.permute_causal_graph(
                intvt_dag,
                permutation_indices=permutation_indices,
            )[0]

            # Permute data
            obs_data, intvn_data = self.permute_data(
                obs_data, intvn_data,
                permutation_indices=permutation_indices,
            )

            # Create mask for valid variables (1 for valid, 0 for padding)
            mask = np.zeros(self.max_variables)
            mask[:current_num_variables] = 1

            # Pad data if necessary
            # This does not do anything if the number of variables is already the maximum
            obs_data = self.pad_data(obs_data, current_num_variables)
            intvn_data = self.pad_data(intvn_data, current_num_variables)
            dag = self.pad_data(dag, current_num_variables)
            intvt_dag = self.pad_data(intvt_dag, current_num_variables)

            # Store data for this batch
            obs_data_array[b] = obs_data
            int_data_array[b] = intvn_data
            causal_graphs_array[b] = dag
            intervened_causal_graphs_array[b] = intvt_dag
            variable_counts[b] = current_num_variables
            masks[b] = mask

        yield (
            obs_data_array,
            int_data_array,
            causal_graphs_array,
            intervened_causal_graphs_array,
            variable_counts,
            masks
        )

    def __iter__(self):
        return iter(self.generate_next_dataset())