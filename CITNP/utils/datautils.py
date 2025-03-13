"""
Data utilities for CITNP.
"""
from typing import List

import numpy as np
import torch as th


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
):
    """
    Sample context and target from dataset
    """
    context_size = sample_context_size(min_context, max_context)
    target_size = sample_size - context_size

    context_idx = np.random.choice(sample_size, context_size, replace=False)
    target_idx = np.random.choice(sample_size, target_size, replace=False)

    context = np.take(obs_data_array, context_idx, axis=1)
    target = np.take(intvn_data_array, target_idx, axis=1)
    return context, target


def standardize_data(data_array: np.ndarray):
    """
    Standardize data.
    """
    mean = np.mean(data_array, axis=(0, 1), keepdims=True)
    std = np.std(data_array, axis=(0, 1), keepdims=True)
    return (data_array - mean) / std


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
        outcome_index[i] = np.random.choice(
            np.delete(np.arange(variable_counts[i]), intvn_index)
        )
    return outcome_index


def transformer_inference_split_withpadding(
    cntxt_split: List[float],
    sample_size: int,
):

    min_context = int(sample_size * cntxt_split[0])
    max_context = int(sample_size * cntxt_split[1])

    def mycollate(batch):
        obs_data_array = batch[0]
        int_data_array = batch[1]
        causal_graphs_array = batch[2]
        intvn_indices = batch[3]
        variable_counts = batch[4]
        masks = batch[5]

        # Standardize the data
        obs_data_array = standardize_data(obs_data_array)
        int_data_array = standardize_data(int_data_array)

        context, target = sample_context_and_target(
            obs_data_array,
            int_data_array,
            sample_size,
            max_context,
            min_context,
        )

        # Randomly sample an outcome index
        outcome_indices = sample_outcome_index(variable_counts, intvn_indices)

        context = th.from_numpy(context).float().unsqueeze(-1)
        target = th.from_numpy(target).float().unsqueeze(-1)
        causal_graphs_array = th.from_numpy(causal_graphs_array).float()

        if masks is not None:
            masks = th.from_numpy(masks).float()

        intvn_indices = th.from_numpy(intvn_indices).long()
        outcome_indices = th.from_numpy(outcome_indices).long()

        return context, target, intvn_indices, outcome_indices, masks, causal_graphs_array

    return mycollate