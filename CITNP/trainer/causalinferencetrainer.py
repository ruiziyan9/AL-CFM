"""
File to train the transformer NP classifier model.
"""

from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch as th
import wandb
from tqdm import tqdm

from CITNP.utils.datautils import transformer_inference_split_withpadding
from CITNP.utils.plotting import plot_predictions_to_wandb
from CITNP.utils.utils import send_to_device


class CausalInferenceTrainer:
    """
    Class to train the causal classifier model.

    Params:
    -------
    - train_dataset: th.utils.data.Dataset
        The training dataset.
    - validation_dataset: th.utils.data.Dataset
        The validation dataset.
    - test_dataset: th.utils.data.Dataset
        The test dataset.
    - model: th.nn.Module
        The model to train.
    - optimizer: th.optim.Optimizer
        The optimizer to use.
    - epochs: int
        The number of epochs to train for.
    - batch_size: int
        The batch size.
    - num_workers: int
        The number of workers to use for the DataLoader.
    - lr_warmup_ratio: float
        The ratio of the total number of steps to use for learning rate warmup.
    - bfloat16: bool
        Whether to use bfloat16 precision.
    - save_dir: Path
        The directory to save the model.
    - cntxt_split: List[float]
        List with min and max of context proportion.
    - scheduler: th.optim.lr_scheduler
        The scheduler to use.
    - use_wandb: bool
        Whether to use wandb.
    - log_step: int
        Logging step.
    - save_checkpoint_every: int
        Number of epochs after which to save checkpoint.
    """

    def __init__(
        self,
        train_dataset: th.utils.data.Dataset,
        validation_dataset: th.utils.data.Dataset,
        test_dataset: th.utils.data.Dataset,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        epochs: int,
        batch_size: int,
        num_workers: int,
        lr_warmup_ratio: float,
        bfloat16: bool,
        save_dir: Path,
        cntxt_split: List[float],
        sample_size: int,
        scheduler: th.optim.lr_scheduler = None,
        use_wandb: bool = True,
        log_step: int = 500,
        save_checkpoint_every: int = 10,
        callback: Optional[List[Callable]] = None,
    ):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr_warmup_ratio = lr_warmup_ratio
        self.bfloat16 = bfloat16
        self.save_dir = save_dir
        self.cntxt_split = cntxt_split
        self.sample_size = sample_size
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        self.log_step = log_step
        self.save_checkpoint_every = save_checkpoint_every
        self.callback = callback

        self.learning_rate = self.optimizer.param_groups[0]["lr"]

        self.initialise_loaders()

    def initialise_loaders(self):
        # For iterable datasets, we don't need to specify the batch size
        if isinstance(self.train_dataset, th.utils.data.IterableDataset):
            self.batch_size = None
            shuffle = False
        else:
            shuffle = True

        collator = partial(
            transformer_inference_split_withpadding,
            cntxt_split=self.cntxt_split,
            sample_size=self.sample_size,
        )
        # Get loaders
        persistent_workers = True if self.num_workers > 0 else False
        self.train_loader = th.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=True),
        )
        self.val_loader = th.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=False),
        )
        self.test_loader = th.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=False),
        )

    def apply_learning_rate_warmup(self, epoch, step, steps_per_epoch, lr_warmup_steps):
        """
        Warmup should be around 10% of the total steps.
        """
        # Compute your global (absolute) step across all epochs.
        global_step = epoch * steps_per_epoch + step

        if global_step < lr_warmup_steps:
            # Linear scale from 0 up to self.learning_rate
            warmup_lr = (global_step / lr_warmup_steps) * self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            # After global_step >= lr_warmup_steps, keep the base (or scheduled) LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def test_single_epoch(
        self,
        test_loader,
        metric_dict,
        calc_metrics=False,
        num_samples=100,
        check_acyclic=False,
    ):
        # with th.no_grad():
        #     self.model.to("cuda")
        #     dtype = th.float32
        #     self.model.eval()
        #     self.model.to(dtype)
        #     all_loss = 0
        #     for i, data in enumerate(tqdm(test_loader, desc="Testing")):
        #         # Get the inputs and targets
        #         inputs, targets, attention_mask = data
        #         targets = targets.to("cuda", dtype=dtype)
        #         inputs = inputs.to("cuda", dtype=dtype)
        #         if attention_mask is not None:
        #             attention_mask = attention_mask.to("cuda", dtype=dtype)
        #         inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / inputs.std(dim=1, keepdim=True)
        #         # Forward pass
        #         adj_logit = self.model(inputs, graph=targets, mask=attention_mask, is_training=False)

        #         if isinstance(adj_logit, tuple):
        #             adj_logit = adj_logit[0]

        #         loss = self.model.calculate_loss(adj_logit, targets)
        #         all_loss += th.sum(loss).cpu().item()
        #         if calc_metrics:
        #             predictions, _ = self.model.sample(
        #                 inputs, num_samples=num_samples, mask=attention_mask
        #             )
        #             auc = auc_graph_scores(targets, predictions)
        #             log_prob = log_prob_graph_scores(targets, predictions.to(targets.device))
        #             e_shd = expected_shd(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy(), check_acyclic=check_acyclic)
        #             e_f1 = expected_f1_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy(), check_acyclic=check_acyclic)
        #             result = {
        #                 "e_shd": list(e_shd),
        #                 "e_f1": list(e_f1),
        #                 "auc": list(auc),
        #                 "log_prob": list(log_prob),
        #             }
        #             if "e_shd" in metric_dict:
        #                 metric_dict["e_shd"] += result["e_shd"]
        #                 metric_dict["e_f1"] += result["e_f1"]
        #                 metric_dict["auc"] += result["auc"]
        #                 metric_dict["log_prob"] += result["log_prob"]
        #             else:
        #                 metric_dict.update(result)
        #     # Log the test loss
        #     loss = all_loss / len(test_loader.dataset)
        #     metric_dict.update(
        #         {
        #             "test_loss": loss,
        #         }
        #     )
        #     dtype = th.bfloat16 if self.bfloat16 else th.float32
        #     self.model.train()
        #     self.model.to(dtype)
        return metric_dict

    def validate_single_epoch(self, val_loader, metric_dict):
        with th.no_grad():
            self.model.eval()
            dtype = th.float32
            self.model.to(dtype)

            all_loss = 0
            for i, data in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                # Get the inputs and targets
                context, target, intvn_indices, outcome_indices, masks, _ = (
                    send_to_device(
                        list(data),
                        device="cuda",
                    )
                )
                context, target = send_to_device(
                    [context, target],
                    device=th.float32,
                )
                if masks is not None:
                    masks = masks.to("cuda", dtype=dtype)

                pred_mean, pred_std, prior_mean, prior_std, z_mean, z_std = self.model(
                    context_data=context,
                    target_data=target,
                    intervention_index=intvn_indices,
                    outcome_index=outcome_indices,
                    variable_mask=masks,
                )
                loss = self.model.calculate_loss(
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                    target=target,
                    outcome_index=outcome_indices,
                    z_mean=z_mean,
                    z_std=z_std,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                )

                all_loss += th.sum(loss).cpu().item()

        if self.use_wandb:
            metric_dict = plot_predictions_to_wandb(
                metric_dict=metric_dict,
                model=self.model,
                loader=val_loader,
                num_plots=10,
                device="cuda",
            )

        # Log the validation loss
        loss = all_loss / len(val_loader.dataset)
        metric_dict.update(
            {
                "val_loss": loss,
            }
        )
        dtype = th.bfloat16 if self.bfloat16 else th.float32
        self.model.train()
        self.model.to(dtype)
        return metric_dict

    def train_single_epoch(
        self,
        train_loader,
        val_loader,
        test_loader,
        epoch,
        lr_warmup_steps,
    ):
        metric_dict = None
        self.model.train()
        dtype = th.bfloat16 if self.bfloat16 else th.float32
        self.model.to(dtype)

        pbar = tqdm(train_loader, desc="Training")
        step_per_epoch = len(train_loader)
        for i, data in enumerate(pbar):
            global_index = epoch * step_per_epoch + i
            # Learning rate warmup
            self.apply_learning_rate_warmup(
                epoch=epoch,
                step=i,
                steps_per_epoch=step_per_epoch,
                lr_warmup_steps=lr_warmup_steps,
            )
            # Get the inputs and targets
            context, target, intvn_indices, outcome_indices, masks, _ = send_to_device(
                list(data),
                device="cuda",
            )
            context, target = send_to_device(
                [context, target],
                device=th.bfloat16 if self.bfloat16 else th.float32,
            )
            if masks is not None:
                masks = masks.to("cuda", dtype=dtype)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            pred_mean, pred_std, prior_mean, prior_std, z_mean, z_std = self.model(
                context_data=context,
                target_data=target,
                intervention_index=intvn_indices,
                outcome_index=outcome_indices,
                variable_mask=masks,
            )
            loss = self.model.calculate_loss(
                pred_mean=pred_mean,
                pred_std=pred_std,
                target=target,
                outcome_index=outcome_indices,
                z_mean=z_mean,
                z_std=z_std,
                prior_mean=prior_mean,
                prior_std=prior_std,
            )
            if isinstance(loss, tuple):
                data_loss, kl = loss
                loss = data_loss + kl
            else:
                data_loss, kl = None, None

            loss.backward()
            # Gradient clipping
            th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Optimize
            self.optimizer.step()
            if global_index % self.log_step == 0:
                metric_dict = {
                    "train loss": loss.mean().item(),
                }
                if data_loss is not None:
                    metric_dict.update({"data loss": data_loss.mean().item()})
                if kl is not None:
                    metric_dict.update({"kl": kl.mean().item()})
                current_lr = self.optimizer.param_groups[0]["lr"]
                metric_dict.update({"learning_rate": current_lr})
                if global_index > 0:
                    metric_dict = self.validate_single_epoch(val_loader, metric_dict)
                    metric_dict = self.test_single_epoch(test_loader, metric_dict)
                    if self.callback is not None:
                        for callback in self.callback:
                            metric_dict = callback(metric_dict)
                if self.use_wandb:
                    wandb.log(metric_dict)
            if (global_index + 1) % self.save_checkpoint_every == 0:
                # Save the model
                self.save_dir.mkdir(parents=True, exist_ok=True)
                th.save(
                    self.model.state_dict(),
                    self.save_dir / "model_{}.pt".format(epoch),
                )
            pbar.set_description(
                "Epoch: {}, Loss: {:.4f}".format(epoch, loss.mean().item())
            )
        # Save the model
        self.save_dir.mkdir(parents=True, exist_ok=True)
        th.save(
            self.model.state_dict(),
            self.save_dir / "model_{}.pt".format(epoch),
        )
        return metric_dict

    def train(self):
        # Set model to train
        self.model.to("cuda")
        # Find the total number of steps for warmup
        lr_warmup_steps = int(
            self.lr_warmup_ratio * len(self.train_loader) * self.epochs
        )
        for epoch in range(self.epochs):
            self.train_single_epoch(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                epoch=epoch,
                lr_warmup_steps=lr_warmup_steps,
            )
            # Step the scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()
        pass
