from typing import Optional

import numpy as np


def normalise_variable(
    data: np.ndarray,
    axis: int,
    return_stats: bool = False,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """
    Normalise data to have mean 0 and variance 1.
    """
    # Skip if data is empty
    if data.size == 0:
        return data

    if mean is None or std is None:
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

    normed_data = (data - mean) / std

    if return_stats:
        return normed_data, mean, std
    else:
        return normed_data


if __name__ == "__main__":
    data = np.random.gamma(1, 1, size=(100, 5))
    print(data.mean(axis=0))
    print(normalise_variable(data, axis=0).mean(axis=0))
