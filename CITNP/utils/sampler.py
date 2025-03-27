from typing import Tuple

import torch
import torch.nn as nn


def init_mean_mlp(m):
    if isinstance(m, nn.Linear):
        # Small weights: mean 0, std ~ 0.1
        nn.init.normal_(m.weight, mean=0.0, std=1e-5)
        nn.init.constant_(m.bias, torch.tensor(0.0))


def init_std_mlp(m):
    if isinstance(m, nn.Linear):
        # Small weights: mean 0, std ~ 0.1
        nn.init.normal_(m.weight, mean=0.0, std=1e-5)
        final_bias = torch.log(torch.exp(torch.tensor(1)) - 1)
        nn.init.constant_(m.bias, final_bias)


class LatentSampler:
    def sample(
        self, context: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainZSampler(LatentSampler):

    def __init__(self, mean_mlp, std_mlp, d_model, num_samples):
        self.mean_mlp = mean_mlp
        self.std_mlp = std_mlp
        self.d_model = d_model
        self.num_samples = num_samples
        init_mean_mlp(self.mean_mlp[-1])
        init_std_mlp(self.std_mlp[-1])

    def sample(self, outcome_yx: torch.Tensor, **kwargs):
        mean = self.mean_mlp(outcome_yx)
        std = self.std_mlp(outcome_yx)
        std = torch.nn.functional.softplus(std)

        z_samples = torch.distributions.Normal(mean, std).rsample((self.num_samples,))
        return z_samples, mean, std


class EvalZSampler(LatentSampler):

    def __init__(self, d_model, num_samples_eval):
        self.d_model = d_model
        self.num_samples_eval = num_samples_eval

    def sample(self, trgt_data: torch.Tensor, **kwargs):
        batch_size, num_trgt = trgt_data.size(0), trgt_data.size(1)
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
