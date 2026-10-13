"""Epoch callback for CITNP's custom trainer (CausalInferenceTrainer).

CausalInferenceTrainer calls on_epoch_begin(trainer, epoch) before it creates
the epoch's training iterator, so set_epoch takes effect for that epoch.
Pass training datasets only; validation/test datasets use
fresh_each_epoch=False and never need set_epoch.
"""

from CITNP.trainer.callbacks import Callback


class SetEpochCallback(Callback):
    """Calls ds.set_epoch(epoch) on every dataset passed in, at on_epoch_begin.

    Needed only with non-persistent DataLoader workers; with persistent workers
    or num_workers=0 the datasets' own counters already give fresh tasks, and
    the call is harmless.
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)

    def on_epoch_begin(self, trainer, epoch):
        for ds in self.datasets:
            ds.set_epoch(epoch)
