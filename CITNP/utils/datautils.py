"""
Data utilities for CITNP.
"""

from typing import List

import h5py
import numpy as np
import torch as th

from CITNP.utils.processing import normalise_variable


def stack_list_of_arrays(list_of_arrays):
    """
    Stack a list of arrays.
    """
    if list_of_arrays[0] is None:
        return None
    return np.stack(list_of_arrays, axis=0)


def sample_context_size(
    min_context: int,
    max_context: int,
):
    return np.random.randint(min_context, max_context + 1)


def sample_context_and_target(
    obs_data_array: np.ndarray,
    intvn_data_array: np.ndarray,
    sample_size: int,
    max_context: int,
    min_context: int,
    is_training: bool = False,
):
    """
    Sample context and target from dataset
    """

    if is_training:
        context_size = sample_context_size(min_context, max_context)
        target_size = sample_size - context_size
        context_idx = np.random.choice(sample_size, context_size, replace=False)
        target_idx = np.random.choice(sample_size, target_size, replace=False)
    else:
        # Deterministic sampling
        context_idx = np.arange(sample_size)
        target_idx = np.arange(sample_size)

    context = np.take(obs_data_array, context_idx, axis=1)
    target = np.take(intvn_data_array, target_idx, axis=1)
    return context, target


def sample_outcome_index(
    variable_counts: np.ndarray,
    intvn_indices: np.ndarray,
):
    """
    Sample an outcome index.
    - It should be different from the intervention index.
    - variable count is the number of variables in each dataset.
    """
    batch_size = variable_counts.shape[0]
    outcome_index = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        intvn_index = intvn_indices[i]
        # outcome_index[i] = np.random.choice(
        #     np.delete(np.arange(variable_counts[i]), intvn_index)
        # )
        outcome_index[i] = np.min(np.delete(np.arange(variable_counts[i]), intvn_index))
        # outcome_index[i] = 1 if intvn_index == 0 else 0
    return outcome_index


def transformer_inference_split_withpadding(
    cntxt_split: List[float],
    sample_size: int,
    is_training: bool = False,
    normalise: bool = True,
    iterable_mode: bool = False,
):
    if is_training:
        min_context = int(sample_size * cntxt_split[0])
        max_context = int(sample_size * cntxt_split[1])
    else:
        min_context = int(sample_size / 2)
        max_context = int(sample_size / 2)

    def mycollate(batch):
        # There are two modes
        if not iterable_mode:
            (
                obs_data_array,
                int_data_array,
                causal_graphs_array,
                intvn_indices,
                variable_counts,
                masks,
            ) = [
                stack_list_of_arrays([sample[i] for sample in batch]) for i in range(6)
            ]
            # Optionally extract the functions if we are returning them
            functions = (
                stack_list_of_arrays([sample[6] for sample in batch])
                if len(batch[0]) > 6
                else None
            )
        else:
            (
                obs_data_array,
                int_data_array,
                causal_graphs_array,
                intvn_indices,
                variable_counts,
                masks,
            ) = batch[:6]

            # Optionally extract the functions if we are returning them
            functions = batch[6] if len(batch) > 6 else None

        # Standardize the data
        if normalise:
            obs_data_array, mean_obs, std_obs = normalise_variable(
                obs_data_array, axis=1, return_stats=True
            )
            int_data_array = normalise_variable(
                int_data_array, axis=1, mean=mean_obs, std=std_obs
            )
            # int_data_array = normalise_variable(
            #     int_data_array,
            #     axis=1,
            # )

        context, target = sample_context_and_target(
            obs_data_array,
            int_data_array,
            sample_size,
            max_context,
            min_context,
            is_training=is_training,
        )

        # Sample an outcome index
        outcome_indices = sample_outcome_index(variable_counts, intvn_indices)

        context = th.from_numpy(context).float().unsqueeze(-1)
        target = th.from_numpy(target).float().unsqueeze(-1)
        causal_graphs_array = th.from_numpy(causal_graphs_array).float()

        if masks is not None:
            masks = th.from_numpy(masks).float()

        intvn_indices = th.from_numpy(intvn_indices).long()
        outcome_indices = th.from_numpy(outcome_indices).long()

        output = (
            context,
            target,
            intvn_indices,
            outcome_indices,
            masks,
            causal_graphs_array,
        )
        if functions is not None:
            output = output + (functions,)
        return output

    return mycollate


class MultipleFileDataset(th.utils.data.Dataset):
    def __init__(self, file_list: list):
        super().__init__()
        self.all_obs_data = []
        self.all_int_data = []
        self.all_graphs = []
        self.all_intvn_indices = []
        self.all_variable_counts = []
        self.all_masks = []
        self.all_functions = []

        for idx, file in enumerate(file_list):
            f = h5py.File(file, "r")
            self.all_obs_data.append(f["obs_data"])
            self.all_int_data.append(f["int_data"])
            self.all_graphs.append(f["causal_graphs"])
            self.all_intvn_indices.append(f["intvn_indices"])
            self.all_variable_counts.append(f["variable_counts"])
            if "masks" in f:
                self.all_masks.append(f["masks"])
            else:
                self.all_masks.append(None)
            if "functions" in f:
                self.all_functions.append(f["functions"])
            else:
                self.all_functions.append(None)
        # Assume all datasets have the same size
        self.size_each_dataset = self.all_obs_data[0].shape[0]

    def load_data(self, data_idx, file_counter):
        obs_data = self.all_obs_data[file_counter][data_idx]
        int_data = self.all_int_data[file_counter][data_idx]
        causal_graph = self.all_graphs[file_counter][data_idx]
        intvn_indices = self.all_intvn_indices[file_counter][data_idx]
        variable_counts = self.all_variable_counts[file_counter][data_idx]

        if self.all_masks[file_counter] is not None:
            masks = self.all_masks[file_counter][data_idx]
        else:
            masks = None

        output = (
            obs_data,
            int_data,
            causal_graph,
            intvn_indices,
            variable_counts,
            masks,
        )
        if self.all_functions[file_counter] is not None:
            functions = self.all_functions[file_counter][data_idx]
            output = output + (functions,)

        yield output

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(data_idx, file_counter))
        return all_data

    def __len__(self):
        return sum([i.shape[0] for i in self.all_obs_data])


class ChunkMultipleFileDataset(th.utils.data.Dataset):
    """
    Dataset for chunked data loading from multiple files.

    We assume that the data is stored in HDF5 files and that each file contains
    multiple datasets. The datasets are assumed to have the same size.

    This class will pick out contiguous chunks of size "batch_size" from
    the datasets. This is useful when there are multiple files with
    different sizes, and we want to load them in chunks.


    Set batch size to 1 in the dataloader to load the data in a single chunk!
    """

    def __init__(self, file_list: list, batch_size: int):
        super().__init__()
        self.all_obs_data = []
        self.all_int_data = []
        self.all_graphs = []
        self.all_intvn_indices = []
        self.all_variable_counts = []
        self.all_masks = []
        self.all_functions = []

        self.batch_size = batch_size

        for idx, file in enumerate(file_list):
            f = h5py.File(file, "r")
            self.all_obs_data.append(f["obs_data"])
            self.all_int_data.append(f["int_data"])
            self.all_graphs.append(f["causal_graphs"])
            self.all_intvn_indices.append(f["intvn_indices"])
            self.all_variable_counts.append(f["variable_counts"])
            if "masks" in f:
                self.all_masks.append(f["masks"])
            else:
                self.all_masks.append(None)
            if "functions" in f:
                self.all_functions.append(f["functions"])
            else:
                self.all_functions.append(None)
        # Assume all datasets have the same size
        self.num_files = len(self.all_obs_data)
        self.size_each_dataset = self.all_obs_data[0].shape[0]
        self.num_chunks = self.size_each_dataset // self.batch_size

        if self.size_each_dataset % self.batch_size != 0:
            raise ValueError(
                "The size of each dataset should be divisible by the batch size."
            )

    def return_chunk(self, array, chunk_idx):
        """
        Return a chunk of the array.
        """
        start_idx = chunk_idx * self.batch_size
        end_idx = (chunk_idx + 1) * self.batch_size
        chunk = array[start_idx:end_idx]
        return chunk

    def load_data(self, chunk_idx, file_counter):
        obs_data = self.all_obs_data[file_counter]
        int_data = self.all_int_data[file_counter]
        causal_graph = self.all_graphs[file_counter]
        intvn_indices = self.all_intvn_indices[file_counter]
        variable_counts = self.all_variable_counts[file_counter]

        if self.all_masks[file_counter] is not None:
            masks = self.all_masks[file_counter]
        else:
            masks = None

        # Get the chunk of data
        obs_data = self.return_chunk(obs_data, chunk_idx)
        int_data = self.return_chunk(int_data, chunk_idx)
        causal_graph = self.return_chunk(causal_graph, chunk_idx)
        intvn_indices = self.return_chunk(intvn_indices, chunk_idx)
        variable_counts = self.return_chunk(variable_counts, chunk_idx)
        if masks is not None:
            masks = self.return_chunk(masks, chunk_idx)

        output = (
            obs_data,
            int_data,
            causal_graph,
            intvn_indices,
            variable_counts,
            masks,
        )
        if self.all_functions[file_counter] is not None:
            functions = self.all_functions[file_counter]
            functions = self.return_chunk(functions, chunk_idx)
            output = output + (functions,)

        yield output

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.num_chunks
        chunk_idx = idx % self.num_chunks

        all_data = next(self.load_data(chunk_idx, file_counter))
        return all_data

    def __len__(self):
        # We want the indices to index the number of files and the number of chunks
        return self.num_files * self.num_chunks
