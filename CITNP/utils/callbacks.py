from typing import List

import torch
import wandb


class GradientChecker:

    def __init__(
        self,
        model: torch.nn.Module,
        watched_layers: List[str],
    ):
        self.model = model
        self.watched_layers = watched_layers

    def __call__(self, metric_dict: dict):

        grad_stats = self.check_grads()
        if self.log_to_wandb:
            wandb.log(grad_stats)

        # Optionally update metrics dict if needed
        metric_dict.update(grad_stats)
        return metric_dict

    def check_grads(
        self,
    ):
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if (self.watched_layers is None) or any(
                    layer in name for layer in self.watched_layers
                ):
                    grad_norm = param.grad.data.norm(2).item()
                    grad_stats[f"grad_norm/{name}"] = grad_norm
        return grad_stats
