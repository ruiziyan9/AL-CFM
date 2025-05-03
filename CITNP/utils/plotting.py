from typing import List

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from CITNP.utils.utils import send_to_device


def to_numpy(data):
    return data.cpu().detach().numpy() if hasattr(data, "cpu") else data


def extract_column(data, idx):
    if data.ndim <= 1:
        raise ValueError("Data must be multi-dimensional.")
    return np.squeeze(data[:, :, idx])


def plot_context_target_predictions(
    context,
    target,
    new_target,
    model_output,
    treatment_idx,
    outcome_index,
    return_fig=False,
    num_mog_samples=15,
):
    # Convert to numpy
    context = to_numpy(context)
    target = to_numpy(target)
    new_target = to_numpy(new_target)
    pred_mean = to_numpy(model_output.pred_mean)
    pred_std = to_numpy(model_output.pred_std)
    # MoG models will also have weights
    if model_output.weights is not None:
        weights = to_numpy(model_output.weights)
    else:
        weights = None

    # Extract relevant columns
    context_treatment = extract_column(context, treatment_idx)
    context_outcome = extract_column(context, outcome_index)
    target_treatment = extract_column(target, treatment_idx)
    target_outcome = extract_column(target, outcome_index)
    new_target_treatment = extract_column(new_target, treatment_idx)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        context_treatment,
        context_outcome,
        color="blue",
        label="Context",
        alpha=0.5,
    )
    ax.scatter(
        target_treatment, target_outcome, color="red", label="Target", alpha=0.5
    )

    # Ensure correct shape for prediction arrays
    if pred_mean.ndim == 2:
        pred_mean = np.expand_dims(pred_mean, axis=0)
        pred_std = np.expand_dims(pred_std, axis=0)

    # Plot predictions
    # If the prediction is not an MoG, plot the mean and std
    if weights is None or pred_mean.shape[-1] == 1:
        n_samples = pred_mean.shape[0]
        for i in range(n_samples):
            mean = np.squeeze(pred_mean[i])
            std = np.squeeze(pred_std[i])
            ax.plot(new_target_treatment, mean, alpha=0.5)
            ax.fill_between(
                new_target_treatment,
                mean - 2 * std,
                mean + 2 * std,
                alpha=0.1 / np.sqrt(n_samples),
            )
    # If the prediction is an MoG, plot samples from MoG
    else:
        # Plot mean prediction.
        mean_weighted = np.squeeze((pred_mean * weights).sum(axis=-1)[0])
        plt.plot(
            new_target_treatment,
            mean_weighted,
            color="green",
            label="Predicted Mean",
            zorder=1,
        )
        # Plot samples.
        for _ in range(num_mog_samples):
            samples = to_numpy(
                sample_MoG(
                    model_output.pred_mean,
                    model_output.pred_std,
                    model_output.weights,
                )
            )
            plt.scatter(
                new_target_treatment,
                samples,
                color="magenta",
                alpha=0.1,
                zorder=5,
            )

    # Final touches
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Outcome")
    ax.set_title("Treatment vs Outcome for Context and Target")
    ax.legend()
    ax.grid(True)

    if return_fig:
        return fig
    else:
        plt.show()
        plt.clf()


def fig_to_wandb_image(fig):
    # Attach a canvas to the figure if it doesn't have one
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image_np = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(
        (height, width, 4)
    )[:, :, :3]
    return wandb.Image(image_np)


def take_single_dataset(list_tensors: List[th.Tensor], idx: int):
    all_tensors = []
    for tensor in list_tensors:
        if tensor is not None:
            all_tensors.append(tensor[idx : idx + 1])
        else:
            all_tensors.append(None)
    return all_tensors


def plot_predictions_to_wandb(
    metric_dict: dict,
    model: th.nn.Module,
    loader: th.utils.data.DataLoader,
    num_plots: int = 10,
    device: str = "cuda",
):
    model.eval()
    model.to(device)
    plots_logged = 0
    wandb_images = []
    with th.no_grad():
        for batch in loader:
            if plots_logged >= num_plots:
                break

            for batch_idx in range(len(batch)):
                if plots_logged >= num_plots:
                    break
                single_batch = take_single_dataset(list(batch), batch_idx)

                context, target, intvn_indices, outcome_indices, masks, graph = (
                    send_to_device(single_batch[:6], device=device)
                )

                num_nodes = context.size(2)
                new_target = (
                    th.linspace(-3, 3, 300)
                    .unsqueeze(1)
                    .repeat(1, num_nodes)[None, :, :, None]
                    .to(device)
                )

                model_output = model(
                    context_data=context,
                    target_data=new_target,
                    intervention_index=intvn_indices,
                    outcome_index=outcome_indices,
                    variable_mask=None,
                )

                fig = plot_context_target_predictions(
                    context=context,
                    target=target,
                    new_target=new_target,
                    model_output=model_output,
                    treatment_idx=intvn_indices,
                    outcome_index=outcome_indices,
                    return_fig=True,
                )

                wandb_images.append(fig_to_wandb_image(fig))

                plt.close(fig)
                plots_logged += 1
    metric_dict["prediction/plots"] = wandb_images
    return metric_dict


def sample_MoG(means, stds, weights):
    """
    Sample from a Mixture of Gaussians (MoG).

    Args:
        means: [B, N, D, K] - Means of Gaussian components
        stds: [B, N, D, K] - Standard deviations
        weights: [B, N, D, K] - Mixture weights (not necessarily normalized)

    Returns:
        samples: [B, N, D] - Sampled values from MoG
    """
    B, N, D, K = means.shape
    assert D == 1, "Only D=1 is supported"

    # Normalize weights
    weights = th.nn.functional.softmax(weights, dim=-1)  # [B, N, D, K]

    # Reshape to [B * N, K] for categorical sampling
    flat_weights = einops.rearrange(weights, "b n d k -> (b n d) k")
    comp_dist = th.distributions.Categorical(probs=flat_weights)
    component_indices = comp_dist.sample()  # [B * N * D]

    # Reshape back to [B, N, D]
    component_indices = einops.rearrange(
        component_indices, "(b n d) -> b n d", b=B, n=N, d=D
    )

    # Use gather to select corresponding means and stds
    comp_indices_exp = component_indices.unsqueeze(-1)  # [B, N, D, 1]
    selected_means = th.gather(means, -1, comp_indices_exp).squeeze(
        -1
    )  # [B, N, D]
    selected_stds = th.gather(stds, -1, comp_indices_exp).squeeze(-1)  # [B, N, D]

    # Sample from standard normal and reparameterize
    eps = th.randn_like(selected_means)
    samples = selected_means + selected_stds * eps  # [B, N, D]

    return samples
