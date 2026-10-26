"""
The following file will contain class of functions that will generate the data
given a causal graph.

All classses will contain the generate_data method, which will generate the
data.

Contains:
- GPLVMFunctionGenerator: Data is generated using a GPLVM.
"""

from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch as th
from scipy.stats import invgamma

from CITNP.datasets.linear_gaussian_prior import (
    LinearGaussianPrior,
    warn_legacy_alias_once,
)
from CITNP.models.nncomponents import ResidualEncoderBlock
from CITNP.utils.gplvm_utils import RBFKernel, SumExpGammaKernels
from CITNP.utils.processing import normalise_variable


class UniformLinear:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents
        self.scale = np.random.gamma(1, 1, size=(1))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a uniform noise function.
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                outputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
                return outputs
            else:
                weight = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[1], 1))
                weight = np.clip(weight, -1, 1)
                feature = np.dot(inputs, weight)
                outputs = feature + np.random.uniform(
                    low=-self.scale, high=self.scale, size=(inputs.shape[0], 1)
                )
                return outputs


class LinearFixedStd:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents
        self.weight = np.random.normal(loc=0, scale=1.0, size=(num_parents, 1))
        self.scale = 0.1

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a uniform noise function.
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                outputs = np.random.normal(
                    loc=0, scale=self.scale, size=(inputs.shape[0], 1)
                )
                return outputs
            else:
                feature = np.dot(inputs, self.weight)
                outputs = feature + np.random.normal(
                    loc=0, scale=self.scale, size=(inputs.shape[0], 1)
                )
                return outputs


class LinearNIGLegacy:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents
        self.var1 = invgamma.rvs(a=3, scale=0.5, size=(1))
        self.var2 = invgamma.rvs(a=3 - 1 / 2, scale=0.5, size=(1))
        self.eta = 1
        self.weight = None
        self.sign = None  # +-1 per edge under the signed-mean prior, else None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a linear model (non-identifiable??).
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                outputs = np.random.normal(
                    loc=0, scale=np.sqrt(self.var2), size=(inputs.shape[0], 1)
                )
                return outputs
            else:
                weight = np.random.normal(
                    loc=0,
                    scale=np.sqrt(self.eta * self.var1),
                    size=(inputs.shape[1], 1),
                )
                # Signed-mean mixture prior: w | s2 ~ N(sign * m, eta * s2),
                # sign = +-1 w.p. 1/2 per edge. Off (m = 0) by default; the
                # signs are drawn after the Gaussian weights and only when
                # m != 0, so the default draws no extra random numbers.
                m = getattr(self, "weight_mean", 0.0)
                if m != 0.0:
                    sign = np.where(
                        np.random.random(size=weight.shape) < 0.5, -1.0, 1.0
                    )
                    weight = weight + m * sign
                    self.sign = sign
                self.weight = weight
                feature = np.dot(inputs, weight)
                outputs = feature + np.random.normal(
                    loc=0, scale=np.sqrt(self.var1), size=(inputs.shape[0], 1)
                )
                return outputs


# Deprecated name of LinearNIGLegacy.
LinearNonIdentifiable = LinearNIGLegacy


class LinearNIG:
    """Linear-Gaussian mechanism with a per-node NIG prior from a
    `LinearGaussianPrior`: s2 ~ IG(prior.node_shape(num_parents), prior.ig_scale),
    w | s2 ~ N(sign * m, eta * s2 * I). Unlike `LinearNIGLegacy` it draws only
    the one variance it needs, and the prior is fixed at construction."""

    def __init__(
        self,
        num_parents: int,
        prior: LinearGaussianPrior,
    ):
        self.num_parents = num_parents
        self.prior = prior
        self.var = invgamma.rvs(
            a=prior.node_shape(num_parents), scale=prior.ig_scale, size=(1)
        )
        self.eta = prior.eta
        self.weight_mean = prior.weight_mean
        self.weight = None
        self.sign = None  # +-1 per edge when weight_mean > 0, else None

    # LinearNIGLegacy names (root / non-root variance); consumers read these.
    @property
    def var1(self) -> np.ndarray:
        """Noise variance (non-root name)."""
        return self.var

    @property
    def var2(self) -> np.ndarray:
        """Noise variance (root name)."""
        return self.var

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a linear model; weights are redrawn on every call.
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                return np.random.normal(
                    loc=0, scale=np.sqrt(self.var), size=(inputs.shape[0], 1)
                )
            weight = np.random.normal(
                loc=0,
                scale=np.sqrt(self.eta * self.var),
                size=(inputs.shape[1], 1),
            )
            if self.weight_mean > 0.0:
                sign = np.where(np.random.random(size=weight.shape) < 0.5, -1.0, 1.0)
                weight = weight + self.weight_mean * sign
                self.sign = sign
            self.weight = weight
            feature = np.dot(inputs, weight)
            return feature + np.random.normal(
                loc=0, scale=np.sqrt(self.var), size=(inputs.shape[0], 1)
            )


# Mechanism classes of the linear-Gaussian NIG families.
LINEAR_NIG_MECHANISMS = (LinearNIGLegacy, LinearNIG)


class SinFunction:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents
        self.scale = np.random.gamma(1, 1, size=(1))
        self.sin_period = np.random.uniform(
            0.5, 10, size=(1)
        )  # Period range from 1 to 6

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a sinusoidal function.
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                outputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
                return outputs
            else:
                outputs = np.sin(
                    (2 * np.pi / self.sin_period) * inputs.sum(axis=-1)
                )  # Proper period scaling
                outputs += np.random.normal(
                    loc=0, scale=self.scale, size=(inputs.shape[0])
                )  # Add noise
                return outputs[:, None]


class GPFunctions:
    """
    Helper class to sample from GPs.

    If no parents are given, we sample from a normal distribution.
    If parents are given, we sample from a GP with a sum of exponential
    gamma kernels.
    """

    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents if num_parents > 0 else 1
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        num_kernels = 2

        lengthscale_values = np.random.lognormal(
            -1, 1, size=(num_kernels, self.num_parents)
        )
        lengthscale_values = np.clip(lengthscale_values, 0.1, 5)
        gamma_values = np.random.uniform(0.1, 1, size=(num_kernels,))
        self.noise_dist = th.distributions.Gamma(1, 10)
        # Define the kernel
        self.kernel = SumExpGammaKernels(
            num_kernels=num_kernels,
            gamma_vals=th.from_numpy(gamma_values).to(self.device),
            lengthscale_vals=th.from_numpy(lengthscale_values).to(self.device),
        )

    def sample_gp(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from an GPLVM.
        """
        with th.no_grad():
            inputs = th.from_numpy(inputs).to(self.device).to(th.float64)
            covariance = self.kernel(inputs, inputs)

            noise_value = (
                self.noise_dist.sample((inputs.shape[0],))
                .to(self.device)
                .to(th.float64)
            )
            covariance = covariance + 1e-4 * th.eye(inputs.shape[0], device=self.device)
            mean = th.zeros(inputs.shape[0], device=self.device)
            normal_dist = th.distributions.MultivariateNormal(mean, covariance)
            output = normal_dist.sample() + noise_value.flatten() * th.randn_like(mean)
            final_output = output.detach().cpu().numpy().astype(np.float32)
            return final_output[:, None]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Samples from GP given kernel and mean.
        """
        input_num = inputs.shape[-1]
        if input_num == 0:
            # If there is no parent, we sample from a normal distribution.
            inputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))

        outputs = self.sample_gp(inputs)
        return outputs


class ResNeuralNetFunctions:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents + 1 if num_parents > 0 else 1
        self.device = th.device("cpu")
        num_blocks = np.random.randint(1, 8)
        hidden_dim = 2 ** np.random.randint(5, 11)
        self.model = th.nn.Sequential(
            *[
                ResidualEncoderBlock(
                    input_dim=(self.num_parents if i == 0 else hidden_dim),
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim if i < num_blocks - 1 else 1,
                    zero_init=False,
                    random_activation=True,
                )
                for i in range(num_blocks)
            ]
        ).to(self.device)
        self.noise_dist = th.distributions.Gamma(1, 10)

        if np.random.rand() > 0.5:
            self.latent_dist = partial(np.random.normal, loc=0, scale=1.0)
        else:
            self.latent_dist = partial(np.random.uniform, low=-1, high=1)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                inputs = self.latent_dist(size=(inputs.shape[0], 1))
            else:
                # If there are parents, append latent.
                latent = self.latent_dist(size=(inputs.shape[0], 1))
                inputs = np.concatenate([inputs, latent], axis=-1)
            inputs = th.from_numpy(inputs).to(self.device).to(th.float32)
            outputs = self.model(inputs).detach()
            noise_scale = (
                self.noise_dist.sample((inputs.shape[0], 1))
                .to(self.device)
                .to(th.float32)
            )
            outputs = (
                (outputs + noise_scale * th.randn_like(outputs))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return outputs


class NeuralNetFunctions:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents + 1 if num_parents > 0 else 1
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.model = th.nn.Sequential(
            th.nn.Linear(self.num_parents, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 1),
        ).to(self.device)

        self.noise_dist = th.distributions.Gamma(1, 10)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                inputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
            else:
                # If there are parents, append latent.
                latent = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
                inputs = np.concatenate([inputs, latent], axis=-1)
            inputs = th.from_numpy(inputs).to(self.device).to(th.float32)
            outputs = self.model(inputs).detach()
            noise_scale = (
                self.noise_dist.sample((inputs.shape[0], 1))
                .to(self.device)
                .to(th.float32)
            )
            outputs = (
                (outputs + noise_scale * th.randn_like(outputs))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return outputs


class SimpleGPFunctions:
    """
    Helper class to sample from GPs.

    If no parents are given, we sample from a normal distribution.
    If parents are given, we sample from a GP with a sum of exponential
    gamma kernels.
    """

    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents if num_parents > 0 else 1
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.dtype = th.float64

        lengthscale_values = np.random.lognormal(-1, 1, size=(self.num_parents))
        lengthscale_values = np.clip(lengthscale_values, 0.1, 5)
        self.noise_dist = th.distributions.Gamma(concentration=1, rate=5)
        # Define the kernel
        self.kernel = RBFKernel(
            lengthscale=th.from_numpy(lengthscale_values)
            .to(self.device)
            .to(th.float32),
        )

    def sample_gp(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from a GP.
        """
        inputs = th.from_numpy(inputs).to(self.device).to(self.dtype)
        covariance = self.kernel(inputs, inputs)
        noise_value = (
            self.noise_dist.sample((inputs.shape[0],)).to(self.device).to(self.dtype)
        )
        covariance = covariance + 1e-4 * th.eye(
            inputs.shape[0], device=self.device, dtype=self.dtype
        )
        mean = th.zeros(inputs.shape[0], device=self.device, dtype=self.dtype)
        normal_dist = th.distributions.MultivariateNormal(mean, covariance)
        noise_sample = normal_dist.sample().to(self.dtype)
        output = noise_sample + noise_value.flatten() * th.randn_like(mean)
        final_output = output.detach().cpu().numpy().astype(np.float32)
        return final_output[:, None]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Samples from GP given kernel and mean.
        """
        with th.no_grad():
            input_num = inputs.shape[-1]
            if input_num == 0:
                # If there is no parent, we sample from a normal distribution.
                inputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))

            outputs = self.sample_gp(inputs)
        return outputs


class SimpleGPLVMFunctions:
    """
    Helper class to sample from GPLVM.

    If no parents are given, we sample from a normal distribution.
    If parents are given, we sample from a GP with a sum of exponential
    gamma kernels.
    """

    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents if num_parents > 0 else 1
        self.device = th.device("cuda")
        self.dtype = th.float64

        # num_kernels = 3

        lengthscale_values = np.random.lognormal(-0.5, 1, size=(self.num_parents + 1))
        lengthscale_values = np.clip(lengthscale_values, 0.1, 5)
        self.noise_dist = th.distributions.Gamma(concentration=1, rate=5)
        # gamma_values = np.random.uniform(0.2, 1, size=(num_kernels,))
        # Define the kernel
        # self.kernel = SumExpGammaKernels(
        #     num_kernels=num_kernels,
        #     gamma_vals=th.from_numpy(gamma_values).to(self.device),
        #     lengthscale_vals=th.from_numpy(lengthscale_values).to(self.device),
        # )
        self.kernel = RBFKernel(
            lengthscale=th.from_numpy(lengthscale_values)
            .to(self.device)
            .to(self.dtype),
        )
        # if np.random.rand() > 0.5:
        self.latent_dist = partial(np.random.normal, loc=0, scale=1.0)
        # else:
        #     self.latent_dist = partial(np.random.uniform, low=-1, high=1)

    def sample_gplvm(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from a GPLVM with improved handling for non-positive definite matrices.
        """
        with th.no_grad():
            if inputs.shape[-1] > 1:
                # Latent as the mean
                mean = th.from_numpy(inputs[:, -1]).to(self.device).to(self.dtype)
            else:
                mean = th.zeros(inputs.shape[0], device=self.device, dtype=self.dtype)

            inputs = th.from_numpy(inputs).to(self.device).to(self.dtype)

            # Initial computation of covariance
            covariance = self.kernel(inputs, inputs)

            # Symmetrize the covariance matrix
            # covariance = 0.5 * (covariance + covariance.T)

            # Initial jitter value
            jitter = 1e-4
            max_attempts = 10

            for attempt in range(max_attempts):
                try:
                    # Add jitter to diagonal
                    jittered_cov = covariance + jitter * th.eye(
                        inputs.shape[0], device=self.device, dtype=self.dtype
                    )

                    # Try to create MultivariateNormal distribution
                    normal_dist = th.distributions.MultivariateNormal(
                        mean, jittered_cov
                    )

                    # If we get here, the matrix is PD
                    noise_value = (
                        self.noise_dist.sample((inputs.shape[0],))
                        .to(self.device)
                        .to(self.dtype)
                    )

                    noise_sample = normal_dist.sample().to(self.dtype)
                    output = noise_sample + noise_value.flatten() * th.randn_like(mean)
                    final_output = output.detach().cpu().numpy().astype(np.float32)

                    return final_output[:, None]

                except (RuntimeError, th.linalg.LinAlgError, ValueError) as e:
                    # If we get an error, increase jitter and try again
                    old_jitter = jitter
                    jitter *= 10  # Increase jitter by an order of magnitude

                    print(
                        f"Attempt {attempt+1}/{max_attempts}: Matrix not PD with jitter={old_jitter}. "
                        f"Increasing to {jitter}. Error: {str(e)}"
                    )

                    # If this is the last attempt, try eigenvalue correction
                    if attempt == max_attempts - 1:
                        try:
                            print("Final attempt: Using eigenvalue correction")
                            eigvals, eigvecs = th.linalg.eigh(covariance)
                            min_eig = th.min(eigvals)

                            if min_eig < 1e-6:
                                # Add the necessary correction to make all eigenvalues positive
                                correction = (1e-6 - min_eig) + jitter
                                jittered_cov = covariance + correction * th.eye(
                                    inputs.shape[0],
                                    device=self.device,
                                    dtype=self.dtype,
                                )

                                normal_dist = th.distributions.MultivariateNormal(
                                    mean, jittered_cov
                                )
                                noise_value = (
                                    self.noise_dist.sample((inputs.shape[0],))
                                    .to(self.device)
                                    .to(self.dtype)
                                )

                                noise_sample = normal_dist.sample().to(self.dtype)
                                output = (
                                    noise_sample
                                    + noise_value.flatten() * th.randn_like(mean)
                                )
                                final_output = (
                                    output.detach().cpu().numpy().astype(np.float32)
                                )

                                return final_output[:, None]
                        except Exception as eigen_error:
                            print(f"Eigenvalue correction failed: {str(eigen_error)}")

            # If all attempts fail, fall back to a diagonal approximation
            print(
                "All attempts to create a PD matrix failed. Falling back to diagonal approximation."
            )
            diag_cov = th.diag(
                th.diag(covariance)
            )  # Extract diagonal and create diagonal matrix
            diag_cov = diag_cov + 1e-6 * th.eye(
                inputs.shape[0], device=self.device, dtype=self.dtype
            )

            normal_dist = th.distributions.MultivariateNormal(mean, diag_cov)

            noise_value = (
                self.noise_dist.sample((inputs.shape[0],))
                .to(self.device)
                .to(self.dtype)
            )

            noise_sample = normal_dist.sample().to(self.dtype)
            output = noise_sample + noise_value.flatten() * th.randn_like(mean)
            final_output = output.detach().cpu().numpy().astype(np.float32)

            return final_output[:, None]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Samples from GP given kernel and mean.
        """
        input_num = inputs.shape[-1]
        if input_num == 0:
            # If there is no parent, we sample from a normal distribution.
            inputs = self.latent_dist(size=(inputs.shape[0], 1))
        else:
            latent = self.latent_dist(size=(inputs.shape[0], 1))
            inputs = np.concatenate([inputs, latent], axis=-1)

        outputs = self.sample_gplvm(inputs)
        return outputs


class DataGenerator(ABC):
    """
    Base class for all causal data generators.
    """

    def __init__(self, num_variables: int, intervention_range_multiplier: float):
        self.number_of_variables = num_variables
        self.intervention_range_multiplier = intervention_range_multiplier

    def _get_inputs(self, parents_of_i: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Get the inputs for the variable i.
        """
        parents_of_i = np.where(parents_of_i)[0]
        inputs = data[:, parents_of_i]
        assert inputs.ndim == 2
        return inputs

    def generate_data(
        self,
        causal_graph: np.ndarray,
        sample_size: int,
        intervention_index: int,
        return_functions: bool = False,
        normalise: bool = True,
    ) -> np.ndarray:
        """
        Generate functions for the SCM.

        For now, we always intervene on the 0th index variable.

        Args:
        ----------
        causal_graph : np.ndarray shape (num_variables, num_variables)
            Causal graph to use for the data generation. Where row i is a parent of column j.
            A[i, j] = 1 if i -> j.

        sample_size : int
            Number of samples to generate.

        intervention_index : int
            Index of the variable to intervene on. The parents of this variable
            will be set to 0.

        return_functions : bool
            Whether to return the functions used to generate the data.
            This is useful in the linear case to compare to analytical result.
        """
        # Functions will be a dict with keys being the variable number
        function_dict = self.generate_functions(causal_graph)
        functions = np.zeros((self.number_of_variables, 3))
        obs_data = np.zeros((sample_size, self.number_of_variables))
        intvn_data = np.zeros((sample_size, self.number_of_variables))
        # Causal graph row i is a parent of column j.
        # The causal graph should be upper triangular.
        # We always need to generate the cause first.
        # Thus, we need to loop over the columns.
        loop_order = np.arange(self.number_of_variables)

        # We need to make sure that the interventions are sampled from the
        # SAME FUNCTION!
        for i in loop_order:
            function_for_i = function_dict[i]
            parents_of_i = causal_graph[:, i]

            # if return_functions:
            #     non_zero_parents = np.where(parents_of_i == 1)[0]
            #     functions[:, i] = function_dict[i].weight[0]

            # Observational data
            # Inputs will be an empty array if there are no parents.
            obs_inputs = self._get_inputs(parents_of_i, obs_data)
            # Intervention on i we set the parents to 0
            if i == intervention_index:
                variable_obs = function_for_i(obs_inputs)
                # variable_intvn = np.random.normal(
                #     loc=np.mean(variable_obs, axis=0),
                #     scale=self.intervention_range_multiplier * np.std(variable_obs, axis=0),
                #     size=(sample_size, 1),
                # )
                variable_intvn = np.random.normal(
                    loc=0,
                    scale=self.intervention_range_multiplier,
                    size=(sample_size, 1),
                )
                if normalise:
                    variable_obs, mean_obs, std_obs = normalise_variable(variable_obs, axis=0, return_stats=True)
                    # variable_intvn = normalise_variable(variable_intvn, axis=0, mean=mean_obs, std=std_obs)
            else:
                intvn_parents_of_i = parents_of_i
                intvn_inputs = self._get_inputs(intvn_parents_of_i, intvn_data)

                full_inputs = np.concatenate([obs_inputs, intvn_inputs], axis=0)

                variable = function_for_i(full_inputs)

                variable_obs = variable[:sample_size]
                variable_intvn = variable[sample_size:]

                if normalise:
                    variable_obs, mean_obs, std_obs = normalise_variable(
                        variable_obs, axis=0, return_stats=True
                    )
                    variable_intvn = normalise_variable(
                        variable_intvn, axis=0, mean=mean_obs, std=std_obs
                    )
            if return_functions:
                if isinstance(function_dict[0], LINEAR_NIG_MECHANISMS):
                    if function_dict[i].weight is not None:
                        functions[i] = (
                            function_dict[i].weight[0][0],
                            function_dict[i].var1[0],
                            function_dict[i].var2[0],
                        )
                    else:
                        functions[i] = (
                            None,
                            function_dict[i].var1[0],
                            function_dict[i].var2[0],
                        )

            obs_data[:, i : i + 1] = variable_obs
            intvn_data[:, i : i + 1] = variable_intvn

        assert not np.isnan(obs_data).any(), "Obs Data contains NaNs!"
        assert not np.isinf(obs_data).any(), "Obs Data contains infs!"
        assert not np.isnan(intvn_data).any(), "Intvn Data contains NaNs!"
        assert not np.isinf(intvn_data).any(), "Intvn Data contains infs!"
        output = (obs_data, intvn_data)
        if return_functions:
            output = output + (functions,)  # type: ignore
        return output

    @abstractmethod
    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        """
        raise NotImplementedError()


class SinusoidFunctionGenerator(DataGenerator):
    """
    Will generate data using sinusoidal functions respecting a given causal graph.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = SinFunction(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class GPFunctionGenerator(DataGenerator):
    """
    Will generate data using Gaussian Process model priors
    respecting a given causal graph.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = GPFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class NeuralNetFunctionGenerator(DataGenerator):
    """
    Will generate data using Neural Net model priors
    respecting a given causal graph.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = NeuralNetFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class SimpleGPFunctionGenerator(DataGenerator):
    """
    Will generate data using Gaussian Process model priors
    respecting a given causal graph. These functions will have a simple
    kernel.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = SimpleGPFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class SimpleGPLVMFunctionGenerator(DataGenerator):
    """
    Will generate data using GPLVM model priors
    respecting a given causal graph. These functions will have a simple
    kernel.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = SimpleGPLVMFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class UniformLinearFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = UniformLinear(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class LinearFixedStdFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = LinearFixedStd(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class LinearNIGLegacyFunctionGenerator(DataGenerator):
    """Legacy NIG prior (registry key `linear_nig_legacy`): roots IG(2.5, 0.5),
    every non-root IG(3.0, 0.5). Not score-equivalent for nodes with two or
    more parents.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function = LinearNIGLegacy(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class LinearNonIdentifiableFunctionGenerator(LinearNIGLegacyFunctionGenerator):
    """Deprecated alias (registry key `linearnonidentifiable`) of
    `LinearNIGLegacyFunctionGenerator`; warns once per process."""

    def __init__(self, *args, **kwargs):
        warn_legacy_alias_once()
        super().__init__(*args, **kwargs)


class LinearNIGFunctionGenerator(DataGenerator):
    """Per-node NIG prior with free hyperparameters (registry key `linear_nig`).
    The prior is given at construction; the default is `LinearGaussianPrior.iw()`.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    prior : LinearGaussianPrior | None
        Per-node prior (default: `LinearGaussianPrior.iw()`).
    """

    def __init__(
        self,
        num_variables: int,
        intervention_range_multiplier: float,
        prior: LinearGaussianPrior | None = None,
    ):
        super().__init__(num_variables, intervention_range_multiplier)
        self.prior = LinearGaussianPrior.iw() if prior is None else prior

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph (one `LinearNIG` per node).
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            num_parents = int(np.sum(causal_graph[:, i]))
            function_dict[i] = LinearNIG(num_parents, self.prior)
        return function_dict


class LinearIWFunctionGenerator(LinearNIGFunctionGenerator):
    """Inverse-Wishart family (registry key `linear_iw`): the prior must be
    IW-compatible, hence score-equivalent."""

    def __init__(
        self,
        num_variables: int,
        intervention_range_multiplier: float,
        prior: LinearGaussianPrior | None = None,
    ):
        super().__init__(num_variables, intervention_range_multiplier, prior)
        if not self.prior.is_iw_compatible:
            raise ValueError(f"linear_iw needs an IW-compatible prior, got {self.prior}")


class NeuralGPLVMFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function: SimpleGPLVMFunctions | NeuralNetFunctions
            # Randomly choose between GPLVM and NeuralNet
            if np.random.rand() > 0.5:
                function = SimpleGPLVMFunctions(num_parents=num_parents)
            else:
                function = NeuralNetFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class NeuralSimpleGPFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function: SimpleGPFunctions | NeuralNetFunctions
            # Randomly choose between simple GP and NeuralNet
            if np.random.rand() > 0.5:
                function = SimpleGPFunctions(num_parents=num_parents)
            else:
                function = NeuralNetFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class ResNeuralFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function: ResNeuralNetFunctions
            # Randomly choose between GPLVM and NeuralNet
            function = ResNeuralNetFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


class ResNeuralGPLVMFunctionGenerator(DataGenerator):
    """

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            num_parents = int(np.sum(parents_of_i))

            function: SimpleGPLVMFunctions | NeuralNetFunctions
            # Randomly choose between GPLVM and NeuralNet
            if np.random.rand() > 0.5:
                function = SimpleGPLVMFunctions(num_parents=num_parents)
            else:
                function = ResNeuralNetFunctions(num_parents=num_parents)
            function_dict[i] = function

        return function_dict
