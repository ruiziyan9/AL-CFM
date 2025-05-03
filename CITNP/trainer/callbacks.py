from dataclasses import asdict

import torch as th
import wandb
from tqdm import tqdm

from CITNP.utils.configs import (
    BaseModelConfig,
    DataConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
)
from CITNP.utils.plotting import plot_predictions_to_wandb


class Callback:
    """Base class for all callbacks."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, epoch_logs):
        pass

    def on_batch_begin(self, trainer, batch_idx):
        pass

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        pass

    def on_validation_begin(self, trainer):
        pass

    def on_validation_end(self, trainer, validation_logs):
        pass

    def on_test_begin(self, trainer):
        pass


class ProgressBarCallback(Callback):
    """Handles TQDM progress bars."""

    def __init__(self):
        self.train_pbar = None
        self.val_pbar = None

    def on_epoch_begin(self, trainer, epoch):
        self.train_pbar = tqdm(
            total=trainer.steps_per_epoch,
            desc=f"Epoch {epoch+1}/{trainer.trainingconfig.epochs} Training",
            leave=True,
        )

    def on_epoch_end(self, trainer, epoch, epoch_logs):
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar = None

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        if self.train_pbar:
            self.train_pbar.update(1)
            self.train_pbar.set_postfix(batch_logs)

    def on_validation_begin(self, trainer):
        self.val_pbar = tqdm(
            total=len(trainer.val_loader), desc="Validation", leave=False
        )

    def on_validation_end(self, trainer, validation_logs):
        if self.val_pbar:
            self.val_pbar.close()
            self.val_pbar = None
        print(f"Validation results: {validation_logs}")


class WandbLoggingCallback(Callback):
    """Logs metrics to Weights & Biases."""

    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_config: LoggingConfig,
        optim_config: OptimizerConfig,
        data_config: DataConfig,
        model_config: BaseModelConfig,
        train_config: TrainingConfig,
    ):
        self.project_name = project_name
        self.run_name = run_name
        self.log_config = log_config
        self.optim_config = optim_config
        self.data_config = data_config
        self.model_config = model_config
        self.trainer_config = train_config
        self._wandb_initialized = False

    def _initialize_wandb(self, trainer):
        if self.log_config.use_wandb and not self._wandb_initialized:

            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config={
                    **asdict(self.data_config),
                    **asdict(self.model_config),
                    **asdict(self.optim_config),
                    **asdict(self.trainer_config),
                    **asdict(self.log_config),
                },
            )
            wandb.watch(
                trainer.model, log="gradients", log_freq=self.log_config.log_step * 10
            )
            self._wandb_initialized = True

    def on_train_begin(self, trainer):
        self._initialize_wandb(trainer)

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        if (
            self.log_config.use_wandb
            and self._wandb_initialized
            and trainer.global_step % self.log_config.log_step == 0
        ):
            # Log learning rate and batch loss
            log_data = {f"train/{k}": v for k, v in batch_logs.items()}
            log_data["train/learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
            wandb.log(log_data, step=trainer.global_step)

    def on_validation_end(self, trainer, validation_logs):
        if self.log_config.use_wandb and self._wandb_initialized:
            log_data = {f"val/{k}": v for k, v in validation_logs.items()}
            if self.log_config.plot_validation_samples:
                plots = plot_predictions_to_wandb(
                    metric_dict={},
                    model=trainer.model,
                    loader=trainer.val_loader,
                    num_plots=self.log_config.num_validation_plots,
                    device=trainer.device,
                )
                log_data.update(plots)
            wandb.log(log_data, step=trainer.global_step)

    def on_train_end(self, trainer):
        if self.log_config.use_wandb and self._wandb_initialized:
            wandb.finish()


class CheckpointCallback(Callback):
    """Saves model checkpoints."""

    def __init__(self, log_config: LoggingConfig):
        self.log_config = log_config
        self.best_val_loss = float("inf")

    def on_train_begin(self, trainer):
        self.log_config.save_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        # Save based on step count
        if (
            trainer.global_step + 1
        ) % self.log_config.save_checkpoint_every_n_steps == 0:
            self._save_checkpoint(trainer, f"step_{trainer.global_step + 1}")

    def on_validation_end(self, trainer, validation_logs):
        # Optionally save best model based on validation loss
        current_val_loss = validation_logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            print(
                f"Validation loss improved ({self.best_val_loss:.4f} -> {current_val_loss:.4f}). Saving best model..."
            )
            self.best_val_loss = current_val_loss
            self._save_checkpoint(trainer, "best_model")

    def on_epoch_end(self, trainer, epoch, epoch_logs):
        pass

    def _save_checkpoint(self, trainer, filename_suffix: str):
        save_path = self.log_config.save_dir / f"model_{filename_suffix}.pt"
        state = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            # Add any other state you want to save (e.g., best_val_loss)
        }
        th.save(state, save_path)
        print(f"Checkpoint saved to {save_path}")

    def on_test_begin(self, trainer):
        # Load the best model for testing
        best_model_path = self.log_config.save_dir / "model_best_model.pt"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            state = th.load(best_model_path, map_location=trainer.device)
            trainer.model.load_state_dict(state["model_state_dict"])
            if trainer.optimizer:
                trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
            if trainer.scheduler and state["scheduler_state_dict"]:
                trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
        else:
            print(f"No best model found at {best_model_path}. Using current model.")


class LRSchedulerCallback(Callback):
    """Handles LR scheduler steps and warmup."""

    def __init__(self, optim_config: OptimizerConfig):
        self.optim_config = optim_config
        self.base_lr = optim_config.learning_rate
        self.warmup_steps = 0  # Will be calculated in on_train_begin

    def on_train_begin(self, trainer):
        if self.optim_config.lr_warmup_ratio > 0:
            self.warmup_steps = int(
                self.optim_config.lr_warmup_ratio * trainer.total_train_steps
            )
        print(f"LR Warmup active for {self.warmup_steps} steps.")

    def on_batch_begin(self, trainer, batch_idx):
        # Apply warmup scaling *before* the optimizer step
        if trainer.global_step < self.warmup_steps:
            lr_scale = (trainer.global_step + 1) / self.warmup_steps  # Linear warmup
            new_lr = self.base_lr * lr_scale
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = new_lr
        elif self.warmup_steps > 0 and trainer.global_step == self.warmup_steps:
            # Ensure base LR is set after warmup if no scheduler is used
            # If a scheduler IS used, it will take over after warmup
            if not trainer.scheduler:
                for param_group in trainer.optimizer.param_groups:
                    param_group["lr"] = self.base_lr
            print(
                f"Warmup complete. LR set to base/scheduler: {trainer.optimizer.param_groups[0]['lr']:.6f}"
            )

    def on_batch_end(self, trainer, batch_idx, batch_logs):
        # Some schedulers step here
        pass

    def on_epoch_end(self, trainer, epoch, epoch_logs):
        # Some schedulers step here
        pass
