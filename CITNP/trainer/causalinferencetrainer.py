"""
File to train the transformer NP classifier model.
"""

from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence

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
        callback: Optional[Sequence[Callable]] = None,
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

        if self.optimizer is not None:
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

    def apply_learning_rate_warmup(
        self, epoch, step, steps_per_epoch, lr_warmup_steps
    ):
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

    def test_single_epoch(self, test_loader, metric_dict):
        with th.no_grad():
            self.model.eval()
            dtype = th.float32
            self.model.to(dtype)

            all_loss = []
            for i, data in enumerate(tqdm(test_loader, desc="Test", leave=False)):
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

                model_output = self.model(
                    context_data=context,
                    target_data=target,
                    intervention_index=intvn_indices,
                    outcome_index=outcome_indices,
                    variable_mask=masks,
                )
                loss = self.model.calculate_loss(
                    model_output=model_output,
                    target=target,
                    outcome_index=outcome_indices,
                    test=True,
                )

                all_loss.append(loss)

        # if self.use_wandb:
        #     metric_dict = plot_predictions_to_wandb(
        #         metric_dict=metric_dict,
        #         model=self.model,
        #         loader=test_loader,
        #         num_plots=10,
        #         device="cuda",
        #     )

        # Log the mean and std of the test loss
        all_loss = th.concat(all_loss)

        metric_dict.update(
            {
                "test_loss_mean": all_loss.mean().cpu().item(),
                "test_loss_std": all_loss.std().cpu().item(),
                # "test_loss_per_dataset": all_loss.cpu().numpy().tolist(),
            }
        )

        return metric_dict, all_loss

    def validate_single_epoch(self, val_loader, metric_dict):
        with th.no_grad():
            self.model.eval()
            dtype = th.float32
            self.model.to(dtype)

            all_loss = 0
            for i, data in enumerate(
                tqdm(val_loader, desc="Validation", leave=False)
            ):
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

                model_output = self.model(
                    context_data=context,
                    target_data=target,
                    intervention_index=intvn_indices,
                    outcome_index=outcome_indices,
                    variable_mask=masks,
                )
                loss = self.model.calculate_loss(
                    model_output=model_output,
                    target=target,
                    outcome_index=outcome_indices,
                )

                all_loss += loss.cpu().item()

        if self.use_wandb:
            metric_dict = plot_predictions_to_wandb(
                metric_dict=metric_dict,
                model=self.model,
                loader=val_loader,
                num_plots=10,
                device="cuda",
            )

        # Log the validation loss
        dataset = val_loader.dataset
        loss = all_loss / len(dataset)
        if isinstance(dataset, th.utils.data.IterableDataset):
            loss /= dataset.batch_size

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
            context, target, intvn_indices, outcome_indices, masks, _ = (
                send_to_device(
                    list(data),
                    device="cuda",
                )
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
            model_output = self.model(
                context_data=context,
                target_data=target,
                intervention_index=intvn_indices,
                outcome_index=outcome_indices,
                variable_mask=masks,
            )
            loss = self.model.calculate_loss(
                model_output=model_output,
                target=target,
                outcome_index=outcome_indices,
            )
            loss.backward()
            # Gradient clipping
            th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Optimize
            self.optimizer.step()
            if global_index % self.log_step == 0:
                metric_dict = {
                    "train loss": loss.mean().item(),
                }
                current_lr = self.optimizer.param_groups[0]["lr"]
                metric_dict.update({"learning_rate": current_lr})
                if global_index > 0:
                    metric_dict = self.validate_single_epoch(
                        val_loader, metric_dict
                    )
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


class CausalInferenceTester(CausalInferenceTrainer):
    def __init__(
        self, test_dataset, model, batch_size, num_workers, sample_size, use_wandb
    ):
        super().__init__(
            train_dataset=test_dataset,  # unused
            validation_dataset=test_dataset,  # unused
            test_dataset=test_dataset,
            model=model,
            optimizer=None,
            epochs=0,
            batch_size=batch_size,
            num_workers=num_workers,
            lr_warmup_ratio=0.0,
            bfloat16=False,
            save_dir=Path("./"),  # unused
            sample_size=sample_size,
            cntxt_split=[0.5, 0.5],  # unused
            use_wandb=use_wandb,
        )

    def test(self):
        return self.test_single_epoch(
            test_loader=self.test_loader,
            metric_dict={},
        )
