from typing import Tuple

import torch


class LatentSampler:
    def sample(
        self, context: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainZSampler(LatentSampler):

    def __init__(self, mlp, d_model, num_samples):
        self.mlp = mlp
        self.d_model = d_model
        self.num_samples = num_samples

    def sample(self, outcome_yx: torch.Tensor, **kwargs):
        z_params = self.mlp(outcome_yx)
        mean, std = z_params.split(self.d_model, dim=-1)
        std = torch.nn.functional.softplus(std)

        z_samples = torch.distributions.Normal(mean, std).rsample((self.num_samples,))
        return z_samples, mean, std


class EvalZSampler(LatentSampler):

    def __init__(self, d_model, num_samples_eval):
        self.d_model = d_model
        self.num_samples_eval = num_samples_eval

    def sample(self, trgt_data: torch.Tensor, **kwargs):
        batch_size, num_trgt = trgt_data.size(0), trgt_data.size(1) - 1
        z_samples = torch.randn(
            self.num_samples_eval,
            batch_size,
            num_trgt,
            self.d_model,
            device=trgt_data.device,
            dtype=trgt_data.dtype,
        )
        mean = torch.zeros_like(z_samples[0:1])
        std = torch.ones_like(z_samples[0:1])
        return z_samples, mean, std
