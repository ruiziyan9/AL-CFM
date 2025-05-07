"""
The following file will contain class of functions that will generate the data
given a causal graph.

All classses will contain the generate_data method, which will generate the
data.

Contains:
- GPLVMFunctionGenerator: Data is generated using a GPLVM.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy.stats import invgamma

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
                outputs = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[0], 1)
                )
                return outputs
            else:
                weight = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[1], 1)
                )
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


class LinearNonIdentifiable:
    def __init__(
        self,
        num_parents: int,
    ):
        self.num_parents = num_parents
        self.var1 = invgamma.rvs(a=3, scale=0.5, size=(1))
        self.var2 = invgamma.rvs(a=3 - 1 / 2, scale=0.5, size=(1))
        self.eta = 1
        self.weight = None

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
                self.weight = weight
                feature = np.dot(inputs, weight)
                outputs = feature + np.random.normal(
                    loc=0, scale=np.sqrt(self.var1), size=(inputs.shape[0], 1)
                )
                return outputs


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
                outputs = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[0], 1)
                )
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
            covariance = covariance + 1e-4 * th.eye(
                inputs.shape[0], device=self.device
            )
            mean = th.zeros(inputs.shape[0], device=self.device)
            normal_dist = th.distributions.MultivariateNormal(mean, covariance)
            output = normal_dist.sample() + noise_value.flatten() * th.randn_like(
                mean
            )
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
                inputs = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[0], 1)
                )
            else:
                # If there are parents, append latent.
                latent = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[0], 1)
                )
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
            self.noise_dist.sample((inputs.shape[0],))
            .to(self.device)
            .to(self.dtype)
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
                inputs = np.random.normal(
                    loc=0, scale=1.0, size=(inputs.shape[0], 1)
                )

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
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.dtype = th.float64

        lengthscale_values = np.random.lognormal(
            -0.5, 1, size=(self.num_parents + 1)
        )
        lengthscale_values = np.clip(lengthscale_values, 0.1, 5)
        self.noise_dist = th.distributions.Gamma(concentration=1, rate=5)
        # Define the kernel
        self.kernel = RBFKernel(
            lengthscale=th.from_numpy(lengthscale_values)
            .to(self.device)
            .to(self.dtype),
        )

    def sample_gplvm(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from a GPLVM.
        """
        with th.no_grad():
            if inputs.shape[-1] > 1:
                # Latent as the mean
                mean = th.from_numpy(inputs[:, -1]).to(self.device).to(self.dtype)
            else:
                mean = th.zeros(
                    inputs.shape[0], device=self.device, dtype=self.dtype
                )

            inputs = th.from_numpy(inputs).to(self.device).to(self.dtype)
            covariance = self.kernel(inputs, inputs)
            noise_value = (
                self.noise_dist.sample((inputs.shape[0],))
                .to(self.device)
                .to(self.dtype)
            )
            covariance = covariance + 1e-4 * th.eye(
                inputs.shape[0], device=self.device, dtype=self.dtype
            )
            normal_dist = th.distributions.MultivariateNormal(mean, covariance)
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
            inputs = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
        else:
            latent = np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0], 1))
            inputs = np.concatenate([inputs, latent], axis=-1)

        outputs = self.sample_gplvm(inputs)
        return outputs


class DataGenerator(ABC):
    """
    Base class for all causal data generators.
    """

    def __init__(
        self,
        num_variables: int,
    ):
        self.number_of_variables = num_variables

    def _get_inputs(
        self, parents_of_i: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
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
                variable_intvn = np.random.normal(
                    loc=0, scale=1.0, size=(sample_size, 1)
                )
                variable_obs = function_for_i(obs_inputs)
                if normalise:
                    obs_inputs = normalise_variable(obs_inputs, axis=0)
                    variable_obs = normalise_variable(variable_obs, axis=0)
            else:
                intvn_parents_of_i = parents_of_i
                intvn_inputs = self._get_inputs(intvn_parents_of_i, intvn_data)

                full_inputs = np.concatenate([obs_inputs, intvn_inputs], axis=0)
                # Pass the inputs to the function
                if normalise:
                    full_inputs = normalise_variable(full_inputs, axis=0)
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
                if isinstance(function_dict[0], LinearNonIdentifiable):
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


class LinearNonIdentifiableFunctionGenerator(DataGenerator):
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

            function = LinearNonIdentifiable(num_parents=num_parents)
            function_dict[i] = function

        return function_dict


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
