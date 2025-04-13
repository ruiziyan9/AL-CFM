import numpy as np


def normalise_variable(data, axis):
    """
    Normalise data to have mean 0 and variance 1.
    """
    # Skip if data is empty
    if data.size == 0:
        return data
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / std


if __name__ == "__main__":
    data = np.random.gamma(1, 1, size=(100, 5))
    print(data.mean(axis=0))
    print(normalise_variable(data, axis=0).mean(axis=0))
