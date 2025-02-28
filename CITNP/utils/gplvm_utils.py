import numpy as np
import torch
import torch.nn as nn


class ExponentialGammaKernel(nn.Module):
    """
    A PyTorch module that computes the Exponential Gamma Kernel between input tensors.
    The kernel function is defined as:
        K(x, y) = exp(-gamma * ||x - y||)
    where ||x - y|| denotes the Euclidean distance between x and y.

    Parameters:
    ----------
    gamma : float
        The scaling parameter of the kernel. Controls the width of the kernel.

    Usage:
    ------
    >>> kernel = ExponentialGammaKernel(gamma=0.5)
    >>> x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> x2 = torch.tensor([[5.0, 6.0]])
    >>> output = kernel(x1, x2)
    """

    def __init__(self, gamma, lengthscale=1.0):
        super(ExponentialGammaKernel, self).__init__()
        self.gamma = gamma
        self.lengthscale = lengthscale
        assert self.gamma > 0, "The gamma parameter must be positive."
        assert self.gamma <= 1, "The gamma parameter must be less than or equal to 1."

    def forward(self, x1, x2):
        """
        Compute the Exponential Gamma Kernel between x1 and x2.

        Parameters:
        ----------
        x1 : torch.Tensor
            Input tensor of shape (n_samples_1, n_features).
        x2 : torch.Tensor
            Input tensor of shape (n_samples_2, n_features).

        Returns:
        -------
        torch.Tensor
            Kernel matrix of shape (n_samples_1, n_samples_2).
        """
        # Compute pairwise Euclidean distances between x1 and x2
        # The cdist function computes the distances efficiently
        distances = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, p=2)  # Euclidean distance (p=2)

        # Symmetrically clip the distances to prevent numerical issues
        distances = torch.clamp(distances, min=1e-36)

        distances = (distances + distances.T) / 2

        # Apply the Exponential Gamma Kernel function
        K = torch.exp(- (distances ** self.gamma))

        return K


class SumExpGammaKernels(nn.Module):

    def __init__(
        self,
        num_kernels: int,
        gamma_vals: np.ndarray,
        lengthscale_vals: np.ndarray,
    ):
        super(SumExpGammaKernels, self).__init__()
        self.kernels = nn.ModuleList(
            [
                ExponentialGammaKernel(gamma=gamma_vals[i], lengthscale=lengthscale_vals[i])
                for i in range(num_kernels)
            ]
        )

    def forward(self, x1, x2):
        K = torch.zeros((x1.shape[0], x2.shape[0]), device=x1.device)
        for kernel in self.kernels:
            K += kernel(x1, x2)
        return K