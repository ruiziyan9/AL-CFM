from typing import List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

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
    pred_mean,
    pred_std,
    treatment_idx,
    outcome_index,
    return_fig=False,
):
    # Convert to numpy
    context = to_numpy(context)
    target = to_numpy(target)
    new_target = to_numpy(new_target)
    pred_mean = to_numpy(pred_mean)
    pred_std = to_numpy(pred_std)

    # Extract relevant columns
    context_treatment = extract_column(context, treatment_idx)
    context_outcome = extract_column(context, outcome_index)
    target_treatment = extract_column(target, treatment_idx)
    target_outcome = extract_column(target, outcome_index)
    new_target_treatment = extract_column(new_target, treatment_idx)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(context_treatment, context_outcome, color="blue", label="Context")
    ax.scatter(target_treatment, target_outcome, color="red", label="Target", alpha=0.5)

    # Ensure correct shape for prediction arrays
    if pred_mean.ndim == 2:
        pred_mean = np.expand_dims(pred_mean, axis=0)
        pred_std = np.expand_dims(pred_std, axis=0)

    # Plot predictions
    n_samples = pred_mean.shape[0]
    for i in range(n_samples):
        mean = np.squeeze(pred_mean[i])
        std = np.squeeze(pred_std[i])
        ax.plot(new_target_treatment, mean, "g-", alpha=0.8)
        ax.fill_between(
            new_target_treatment,
            mean - 2 * std,
            mean + 2 * std,
            color="green",
            alpha=0.2 / np.sqrt(n_samples),
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
                    send_to_device(single_batch, device=device)
                )

                num_nodes = context.size(2)
                new_target = (
                    th.linspace(-3, 3, 300)
                    .unsqueeze(1)
                    .repeat(1, num_nodes)[None, :, :, None]
                    .to(device)
                )

                mean, std, _, _ = model(
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
                    pred_mean=mean,
                    pred_std=std,
                    treatment_idx=intvn_indices,
                    outcome_index=outcome_indices,
                    return_fig=True,
                )

                wandb_images.append(fig_to_wandb_image(fig))

                plt.close(fig)
                plots_logged += 1
    metric_dict["prediction/plots"] = wandb_images
    return metric_dict
