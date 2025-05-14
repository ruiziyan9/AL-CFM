"""
This file contains dataset generators with queue-based prefetching.
"""

import os
import queue
import random
import threading
import time
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import trange

from CITNP.datasets.causal_graph_generator import generate_synthetic_dag
from CITNP.datasets.functions_generator import (
    GPFunctionGenerator,
    LinearFixedStdFunctionGenerator,
    LinearNonIdentifiableFunctionGenerator,
    NeuralGPLVMFunctionGenerator,
    NeuralNetFunctionGenerator,
    NeuralSimpleGPFunctionGenerator,
    ResNeuralFunctionGenerator,
    ResNeuralGPLVMFunctionGenerator,
    SimpleGPFunctionGenerator,
    SimpleGPLVMFunctionGenerator,
    SinusoidFunctionGenerator,
    UniformLinearFunctionGenerator,
)


class BaseInterventionDatasetGenerator(IterableDataset):
    """Base class for interventional datasets."""

    def __init__(
        self,
        function_generator: str,
        num_variables: Union[int, List[int]],
        sample_size: int,
        batch_size: int,
        graph_type: List[str],
        graph_degrees: List[int] | dict[int, List[int]],
        iterations_per_epoch: Optional[int] = 1000,
        return_functions: Optional[bool] = False,
        normalise: Optional[bool] = True,
        show_progress: Optional[bool] = False,
        same_variablenum_per_batch: Optional[bool] = False,
    ):
        # Initialize parameters
        self.function_generator = function_generator
        self.num_variables = num_variables
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.graph_type = graph_type
        self.graph_degrees = graph_degrees
        self.max_num_variables = (
            max(num_variables) if isinstance(num_variables, list) else num_variables
        )
        self.iterations_per_epoch = iterations_per_epoch
        self.return_functions = return_functions
        self.normalise = normalise
        self.show_progress = show_progress
        self.same_variablenum_per_batch = same_variablenum_per_batch

        # Set up function generator
        valid_generators = {
            "gp": GPFunctionGenerator,
            "sinusoid": SinusoidFunctionGenerator,
            "neuralnet": NeuralNetFunctionGenerator,
            "simplegp": SimpleGPFunctionGenerator,
            "simplegplvm": SimpleGPLVMFunctionGenerator,
            "uniformlinear": UniformLinearFunctionGenerator,
            "linearfixedstd": LinearFixedStdFunctionGenerator,
            "linearnonidentifiable": LinearNonIdentifiableFunctionGenerator,
            "neuralgplvm": NeuralGPLVMFunctionGenerator,
            "neuralsimplegp": NeuralSimpleGPFunctionGenerator,
            "resnet": ResNeuralFunctionGenerator,
            "resnetgplvm": ResNeuralGPLVMFunctionGenerator,
        }

        if self.function_generator in valid_generators:
            self.data_generator = partial(valid_generators[self.function_generator])
        else:
            valid_types = ", ".join(valid_generators.keys())
            raise ValueError(
                f"{self.function_generator} is not a valid function generator. Valid options are: {valid_types}."
            )

    def _get_num_variables(self):
        if isinstance(self.num_variables, list):
            return np.random.choice(self.num_variables)
        else:
            return self.num_variables

    def permute_data(self, *args, permutation_indices: np.ndarray) -> list:
        """
        Permute the data to randomise the causal graph.
        """
        permuted_data = [data[:, permutation_indices] for data in args]
        return permuted_data

    def permute_causal_graph(self, *args, permutation_indices: np.ndarray) -> list:
        """
        Permutes the causal graph.
        """
        permuted_causal_graphs = [
            dag[permutation_indices, :][:, permutation_indices] for dag in args
        ]
        return permuted_causal_graphs

    def sample_uniform_expected_degree(self, current_num_variables: int) -> int:
        """
        Sample a degree from the list of expected degrees.
        """
        if isinstance(self.graph_degrees, dict):
            possible_degrees = self.graph_degrees.get(
                current_num_variables, self.graph_degrees
            )
        else:
            possible_degrees = self.graph_degrees
        degree = np.random.choice(possible_degrees)
        return degree

    def randomly_intervene_on_dag(self, dag: np.ndarray) -> int:
        """
        Randomly intervene on a variable in the DAG.
        """
        intervention_index = np.random.choice(dag.shape[0])
        return intervention_index

    def pad_data(
        self, data: np.ndarray, current_vars: int, max_size: int
    ) -> np.ndarray:
        """
        Pad data to match the maximum number of variables.
        """
        if current_vars == max_size:
            return data

        pad_width: Union[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        ]

        # For 2D data: samples x variables
        if data.ndim == 2:
            pad_width = ((0, 0), (0, max_size - current_vars))
        # For 3D data: variables x variables (adjacency matrix)
        elif data.ndim == 3:
            pad_width = (
                (0, 0),
                (0, max_size - current_vars),
                (0, max_size - current_vars),
            )

        return np.pad(data, pad_width, mode="constant", constant_values=0)

    def generate_next_dataset(self):
        """
        Generate the next dataset batch.
        """
        if self.same_variablenum_per_batch:
            current_num_variables = self._get_num_variables()
            max_size = current_num_variables
        else:
            max_size = self.max_num_variables

        # Arrays to store data for each batch
        obs_data_array = np.zeros((self.batch_size, self.sample_size, max_size))
        int_data_array = np.zeros((self.batch_size, self.sample_size, max_size))
        causal_graphs_array = np.zeros((self.batch_size, max_size, max_size))
        functions_array = np.zeros((self.batch_size, max_size, 3))
        intvn_indices = np.zeros(self.batch_size, dtype=int)
        variable_counts = np.zeros(self.batch_size, dtype=int)
        masks = np.zeros((self.batch_size, max_size))

        # Generate each dataset in the batch
        for b in range(self.batch_size):
            # Sample number of variables
            if not self.same_variablenum_per_batch:
                current_num_variables = self._get_num_variables()

            # Sample graph type and expected degree
            expected_node_degree = self.sample_uniform_expected_degree(
                current_num_variables=current_num_variables
            )
            curr_graph_type = np.random.choice(self.graph_type)

            # Generate the causal graph
            dag = generate_synthetic_dag(
                d=current_num_variables,
                s0=expected_node_degree,
                graph_type=curr_graph_type,
            )

            # Generate data
            current_data_generator = self.data_generator(
                num_variables=current_num_variables,
            )

            # Create intervention
            intvn_index = self.randomly_intervene_on_dag(dag)

            # Generate data for this graph
            data = current_data_generator.generate_data(
                causal_graph=dag,
                sample_size=self.sample_size,
                intervention_index=intvn_index,
                return_functions=self.return_functions,
                normalise=self.normalise,
            )

            # Unpack data
            if self.return_functions:
                obs_data, intvn_data, functions = data
            else:
                obs_data, intvn_data = data

            # Permute the data and graphs
            permutation_indices = np.random.permutation(current_num_variables)
            dag = self.permute_causal_graph(
                dag,
                permutation_indices=permutation_indices,
            )[0]

            # Find the index of the intervened variable post permutation
            intvn_index = np.where(permutation_indices == intvn_index)[0][0]

            # Permute data
            obs_data, intvn_data = self.permute_data(
                obs_data,
                intvn_data,
                permutation_indices=permutation_indices,
            )

            # Create mask for valid variables
            mask = np.zeros(max_size)
            mask[:current_num_variables] = 1

            # Pad data if necessary
            obs_data = self.pad_data(obs_data, current_num_variables, max_size=max_size)
            intvn_data = self.pad_data(
                intvn_data, current_num_variables, max_size=max_size
            )
            dag = self.pad_data(dag, current_num_variables, max_size=max_size)

            # Store data for this dataset
            obs_data_array[b] = obs_data
            int_data_array[b] = intvn_data
            causal_graphs_array[b] = dag
            intvn_indices[b] = intvn_index
            variable_counts[b] = current_num_variables
            masks[b] = mask

            if self.return_functions:
                functions = functions[permutation_indices]
                functions_array[b] = functions

        # If num_variables is not a list, masks will be None
        if not isinstance(self.num_variables, list):
            masks = None

        output = (
            obs_data_array,
            int_data_array,
            causal_graphs_array,
            intvn_indices,
            variable_counts,
            masks,
        )

        if self.return_functions:
            output += (functions_array,)

        yield output


class QueuedInterventionDatasetGenerator(BaseInterventionDatasetGenerator):
    """Generate datasets using a function generator with queue-based prefetching.
    This generates observational data along with interventional datasets in a background
    thread pool to improve performance.

    Args:
    ----------
    function_generator : str
        The function generator to use.

    num_variables : Union[int, List[int]]
        The number of variables in the dataset. If a list is provided,
        the number of variables will be randomly sampled from this list for each dataset.

    sample_size : int
        The maximum number of samples to generate.

    batch_size : int
        The number of datasets to generate per batch.

    graph_type : List[str]
        The type of graph to generate.
        - "ER": Erdos-Renyi
        - "SF": Scale-free, Barabasi-Albert

    graph_degrees : List[int]
        The expected degrees of the graph.

    iterations_per_epoch : Optional[int] = 1000
        Number of iterations per epoch.

    prefetch_factor : Optional[int] = 2
        Number of batches to prefetch per worker.

    queue_workers : Optional[int] = 4
        Number of worker threads for data generation.
    """

    def __init__(
        self,
        function_generator: str,
        num_variables: Union[int, List[int]],
        sample_size: int,
        batch_size: int,
        graph_type: List[str],
        graph_degrees: List[int] | dict[int, List[int]],
        iterations_per_epoch: Optional[int] = 1000,
        return_functions: Optional[bool] = False,
        normalise: Optional[bool] = True,
        show_progress: Optional[bool] = False,
        same_variablenum_per_batch: Optional[bool] = False,
        prefetch_factor: Optional[int] = 3,
        queue_workers: Optional[int] = 2,
    ):
        super().__init__(
            function_generator=function_generator,
            num_variables=num_variables,
            sample_size=sample_size,
            batch_size=batch_size,
            graph_type=graph_type,
            graph_degrees=graph_degrees,
            iterations_per_epoch=iterations_per_epoch,
            return_functions=return_functions,
            normalise=normalise,
            show_progress=show_progress,
            same_variablenum_per_batch=same_variablenum_per_batch,
        )

        # Queue parameters
        self.prefetch_factor = prefetch_factor
        self.queue_workers = queue_workers

        # Flag to indicate if queue is initialized
        self.queue_initialized = False
        self.worker_id = None
        self.worker_count = None

    def _initialize_queue(self):
        """Initialize the prefetching queue and worker threads.
        This is now called in __iter__ to ensure it happens in the worker process."""
        if self.queue_initialized:
            return

        # Queue size is based on number of workers and prefetch factor
        self.queue_size = self.queue_workers * self.prefetch_factor
        self.data_queue = queue.Queue(maxsize=self.queue_size)

        # Synchronization
        self.stop_event = threading.Event()
        self.workers = []

        # Start worker threads (only in the process where this is called)
        for worker_id in range(self.queue_workers):
            worker = threading.Thread(
                target=self._data_generation_worker,
                args=(
                    worker_id + (self.worker_id or 0) * 1000,
                ),  # Ensure unique seeds across processes
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        self.queue_initialized = True

    def _data_generation_worker(self, worker_id):
        """Worker thread that generates data and adds it to the queue."""
        # Set worker-specific random seed
        np.random.seed(42 + 1000 + worker_id)

        while not self.stop_event.is_set():
            try:
                # Generate a batch
                batch_data = next(self.generate_next_dataset())

                # Try to add to queue with timeout
                self.data_queue.put(batch_data, timeout=1.0)
            except queue.Full:
                # Queue is full, wait briefly and retry
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"Error in data generation worker {worker_id}: {e}")
                # Add a sentinel to avoid deadlocking the training process
                if not self.stop_event.is_set():
                    try:
                        self.data_queue.put(None, timeout=1.0)
                    except queue.Full:
                        pass
                break

    def __iter__(self):
        """Return an iterator for the dataset."""
        # Get worker info from PyTorch DataLoader
        worker_info = torch.utils.data.get_worker_info()

        # Set worker info for proper thread initialization
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.worker_count = worker_info.num_workers
        else:
            self.worker_id = 0
            self.worker_count = 1

        # Initialize the queue here to ensure it happens in the worker process
        self._initialize_queue()

        # Calculate iterations for this worker
        iterations = self.iterations_per_epoch
        if worker_info is not None:
            # Divide iterations among DataLoader workers
            iterations = self.iterations_per_epoch // worker_info.num_workers
            # Handle remainder if needed
            if worker_info.id < self.iterations_per_epoch % worker_info.num_workers:
                iterations += 1

        # Generate data for assigned iterations
        for _ in range(iterations):
            try:
                # Get batch from queue with timeout to prevent hanging
                batch = self.data_queue.get(timeout=60.0)  # 1-minute timeout

                # Handle potential None sentinel
                if batch is None:
                    raise RuntimeError("Data generation error occurred")

                yield batch

                # Signal the queue that the item has been processed
                self.data_queue.task_done()
            except queue.Empty:
                print(
                    f"Queue timeout occurred in worker {self.worker_id}. Reinitializing queue..."
                )
                # Shutdown existing queue
                self.shutdown()
                # Reinitialize queue
                self.queue_initialized = False
                self._initialize_queue()
                # Try again after reinitialization
                batch = self.data_queue.get(timeout=60.0)
                if batch is None:
                    raise RuntimeError(
                        "Data generation error occurred after reinitialization"
                    )
                yield batch
                self.data_queue.task_done()

    def __len__(self):
        """Return the length of the dataset."""
        return self.iterations_per_epoch

    def shutdown(self):
        """Clean up resources and stop workers."""
        if hasattr(self, "stop_event") and self.stop_event is not None:
            self.stop_event.set()
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=2.0)
            self.queue_initialized = False

    def __del__(self):
        """Ensure resources are properly released when the object is deleted."""
        self.shutdown()


class InterventionDatasetGenerator(BaseInterventionDatasetGenerator):
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
        graph_degrees: List[int] | dict[int, List[int]],
        iterations_per_epoch: Optional[int] = 1000,
        return_functions: Optional[bool] = False,
        normalise: Optional[bool] = True,
        show_progress: Optional[bool] = False,
        same_variablenum_per_batch: Optional[bool] = False,
    ):
        super().__init__(
            function_generator=function_generator,
            num_variables=num_variables,
            sample_size=sample_size,
            batch_size=batch_size,
            graph_type=graph_type,
            graph_degrees=graph_degrees,
            iterations_per_epoch=iterations_per_epoch,
            return_functions=return_functions,
            normalise=normalise,
            show_progress=show_progress,
            same_variablenum_per_batch=same_variablenum_per_batch,
        )

    def __iter__(self):
        # Get worker info for proper sharding
        worker_info = torch.utils.data.get_worker_info()

        # If no worker info or single worker, use the parent's iterator
        if worker_info is None:
            iterations = self.iterations_per_epoch
            worker_id = 0
            num_workers = 1
        else:
            # Partition iterations among workers
            iterations = self.iterations_per_epoch // worker_info.num_workers
            # Handle remainder if needed
            if worker_info.id < self.iterations_per_epoch % worker_info.num_workers:
                iterations += 1
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Set different random seed for each worker to ensure diversity
        random_seed = int(time.time()) + os.getpid() + random.randint(0, 10000)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Generate data for assigned iterations
        for _ in range(iterations):
            yield next(self.generate_next_dataset())

    def __len__(self):
        return self.iterations_per_epoch
