"""
File to train the transformer NP classifier model.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch as th
from tqdm import tqdm

from CITNP.trainer.callbacks import Callback
from CITNP.utils.configs import (
    DataConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
)
from CITNP.utils.datautils import (
    ChunkMultipleFileDataset,
    transformer_inference_split_withpadding,
)
from CITNP.utils.utils import get_causal_direction, send_to_device


class CausalInferenceTrainer:
    """
    Class to train the causal classifier model.

    Params:
    -------
    - model: th.nn.Module
        The model to train.
    """

    def __init__(
        self,
        train_dataset: th.utils.data.Dataset,
        validation_dataset: th.utils.data.Dataset,
        test_dataset: th.utils.data.Dataset,
        model: th.nn.Module,
        dataconfig: DataConfig,
        optimizerconfig: OptimizerConfig,
        trainingconfig: TrainingConfig,
        loggingconfig: LoggingConfig,
        callbacks: Optional[Sequence[Callback]] = None,
    ):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.model = model

        self.dataconfig = dataconfig
        self.optimizerconfig = optimizerconfig
        self.trainingconfig = trainingconfig
        self.loggingconfig = loggingconfig

        self.device = th.device(self.trainingconfig.device)
        self.optimizer = self.optimizerconfig.optimizer
        self.scheduler = self.optimizerconfig.scheduler

        self.train_dtype = getattr(th, self.dataconfig.train_dtype)
        self.eval_dtype = getattr(th, self.dataconfig.eval_dtype)

        self.normalise = dataconfig.normalise
        self.initialise_loaders()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.total_train_steps = self.trainingconfig.epochs * len(self.train_loader)
        self.steps_per_epoch = len(self.train_loader)

        self.callbacks = self._initialise_default_callbacks(callbacks)

    def _initialise_default_callbacks(self, callbacks):
        return callbacks

    def initialise_loaders(self):
        # For iterable datasets, we don't need to specify the batch size
        if isinstance(self.train_dataset, th.utils.data.IterableDataset):
            batch_size = None
            shuffle = False
        elif isinstance(self.train_dataset, ChunkMultipleFileDataset):
            batch_size = 1
            shuffle = True
        else:
            shuffle = True
            batch_size = self.dataconfig.batch_size

        collator = partial(
            transformer_inference_split_withpadding,
            cntxt_split=self.dataconfig.cntxt_split,
            sample_size=self.dataconfig.sample_size,
            normalise=self.normalise,
        )
        # Get loaders
        persistent_workers = True if self.dataconfig.num_workers > 0 else False
        self.train_loader = th.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataconfig.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=True),
        )
        self.val_loader = th.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.dataconfig.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=False),
        )
        self.test_loader = th.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.dataconfig.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=collator(is_training=False),
        )

    def _run_callbacks(self, hook_name: str, *args, **kwargs):
        """Helper to run a specific hook on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)
            if method:
                method(self, *args, **kwargs)

    def _prepare_batch(self, batch: Tuple, dtype: th.dtype) -> Dict[str, th.Tensor]:
        """Sends batch data to the correct device and converts type."""
        context, target, intvn_indices, outcome_indices, masks, graph = send_to_device(
            list(batch[:6]), self.device
        )

        # Convert relevant tensors to the training dtype
        context = context.to(dtype=dtype)
        target = target.to(dtype=dtype)
        if masks is not None:
            masks = masks.to(dtype=dtype)

        return {
            "context": context,
            "target": target,
            "intervention_index": intvn_indices,
            "outcome_index": outcome_indices,
            "variable_mask": masks,
            "causal_graphs": graph,
        }

    def _forward_pass(
        self, batch: Dict[str, th.Tensor], test: Optional[bool] = False
    ) -> Dict[str, Any]:
        model_output = self.model(
            context_data=batch["context"],
            target_data=batch["target"],
            intervention_index=batch["intervention_index"],
            outcome_index=batch["outcome_index"],
            variable_mask=batch["variable_mask"],
        )
        # Assumes model.calculate_loss exists and takes specific args
        loss = self.model.calculate_loss(
            model_output=model_output,
            target=batch["target"],  # Pass the original target data if needed
            outcome_index=batch["outcome_index"],
            test=test,
        )
        return {"loss": loss, "model_output": model_output}

    def train_step(self, batch: Dict[str, th.Tensor]):
        """One step of training."""
        self.model.train()

        outputs = self._forward_pass(batch)
        loss = outputs["loss"]
        self.optimizer.zero_grad()
        loss.backward()

        if self.trainingconfig.gradient_clip_val is not None:
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.trainingconfig.gradient_clip_val
            )

        self.optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch: Dict[str, th.Tensor]) -> Dict[str, float]:
        """One step of validation."""
        self.model.eval()  # Ensure model is in eval mode
        with th.no_grad():  # No gradients needed
            outputs = self._forward_pass(
                batch
            )  # Use original batch, autocast handles types
            loss = outputs["loss"]

        return {"val_loss": loss.item()}  # Return batch validation metrics

    def train_epoch(self, epoch: int):
        """Runs the training loop for one epoch."""
        self._run_callbacks("on_epoch_begin", epoch=epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            self._run_callbacks("on_batch_begin", batch_idx=batch_idx)

            prepared_batch = self._prepare_batch(batch, dtype=self.train_dtype)
            batch_logs = self.train_step(prepared_batch)

            self.global_step += 1
            self._run_callbacks(
                "on_batch_end", batch_idx=batch_idx, batch_logs=batch_logs
            )

            # Figure out whether to run validation
            run_validation = False
            if self.global_step % self.loggingconfig.log_step == 0:
                run_validation = True
            if batch_idx == self.steps_per_epoch - 1:
                run_validation = True

            if run_validation:
                _ = self.validate()

        self._run_callbacks("on_epoch_end", epoch=epoch, epoch_logs=None)

    def validate(self) -> Dict[str, float]:
        """Runs the validation loop."""
        self._run_callbacks("on_validation_begin")
        self.model.to(self.eval_dtype)
        all_val_losses = []

        for batch_idx, batch in enumerate(self.val_loader):
            prepared_batch = self._prepare_batch(batch, dtype=self.eval_dtype)
            batch_val_logs = self.validation_step(prepared_batch)
            all_val_losses.append(batch_val_logs["val_loss"])

        avg_val_loss = sum(all_val_losses) / len(all_val_losses)
        validation_logs = {"val_loss": avg_val_loss}
        self._run_callbacks("on_validation_end", validation_logs=validation_logs)

        self.model.to(self.train_dtype)  # Switch back to training dtype
        return validation_logs

    def train(self):
        """Starts the main training loop."""
        print(
            f"Starting training for {self.trainingconfig.epochs} epochs on {self.device}..."
        )
        self.model.to(self.device)
        self.model.to(self.train_dtype)
        self._run_callbacks("on_train_begin")

        try:
            for epoch in range(self.current_epoch, self.trainingconfig.epochs):
                self.current_epoch = epoch
                self.train_epoch(epoch)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            self._run_callbacks("on_train_end")
            print("Training finished.")

    def test_epoch(self, test_loader, metric_dict):
        with th.no_grad():
            self.model.eval()
            self.model.to(self.eval_dtype)
            self.model.to(self.device)
            losses = defaultdict(list)
            for i, batch in enumerate(tqdm(test_loader, desc="Test", leave=False)):
                # Get the inputs and targets

                prepared_batch = self._prepare_batch(batch, dtype=self.eval_dtype)

                directions = get_causal_direction(
                    graphs=prepared_batch["causal_graphs"],
                    intvn_indices=prepared_batch["intervention_index"],
                    outcome_indices=prepared_batch["outcome_index"],
                )

                output = self._forward_pass(prepared_batch, test=True)

                loss = output["loss"]

                for direction, loss_val in zip(directions, loss):
                    val = loss_val.item()
                    losses[direction].append(val)
                    losses["all"].append(val)

        # Convert to numpy arrays for safe aggregation
        losses_np = {k: np.array(v) for k, v in losses.items()}

        # Aggregate results
        for key in ["all", "downstream", "upstream", "independent"]:
            if key in losses_np and len(losses_np[key]) > 0:
                metric_dict[f"test_loss_{key}_mean"] = losses_np[key].mean()
                metric_dict[f"test_loss_{key}_std"] = losses_np[key].std()
            else:
                metric_dict[f"test_loss_{key}_mean"] = None
                metric_dict[f"test_loss_{key}_std"] = None

        return metric_dict

    def test(self, checkpoint_name="last_model"):
        self._run_callbacks("on_test_begin", checkpoint_name=checkpoint_name)
        return self.test_epoch(
            test_loader=self.test_loader,
            metric_dict={},
        )
