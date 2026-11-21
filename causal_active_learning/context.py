"""The experimental context: rows on the raw scale and which cells were set by
an intervention, and appending acquired (or fantasy) rows to it."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Smallest standard deviation of the observational statistics.
MINIMUM_STANDARD_DEVIATION = 1e-6


@dataclass(frozen=True)
class ExperimentContext:
    """One context on the raw data scale.

    Attributes:
    -----------
    - values: [N, D] float64, finite.
    - intervened: [N, D] bool, True where the cell was set by do(); at most one
      intervened node per row (the current model contract).

    D is read from the tensors and never passed separately.
    """

    values: torch.Tensor
    intervened: torch.Tensor

    def __post_init__(self):
        values = torch.as_tensor(self.values, dtype=torch.float64).cpu()
        intervened = torch.as_tensor(self.intervened, dtype=torch.bool).cpu()
        if values.ndim != 2 or values.shape != intervened.shape:
            raise ValueError("values and intervened must both be [N, D]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("context values must be finite")
        if bool((intervened.sum(-1) > 1).any()):
            raise ValueError(
                "a context row may be intervened on one node at most"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "intervened", intervened)

    @property
    def num_rows(self) -> int:
        """N."""
        return int(self.values.shape[0])

    @property
    def num_variables(self) -> int:
        """D."""
        return int(self.values.shape[1])

    def append(
        self, new_rows: torch.Tensor, target_node: int | None
    ) -> "ExperimentContext":
        """A new context with `new_rows` [R, D] appended, all flagged as
        intervened on `target_node` only (on nothing for None: observational
        rows); this one is unchanged."""
        values, intervened = append_rows_to_context(
            self.values, self.intervened, new_rows, target_node
        )
        return ExperimentContext(values, intervened)

    def observational_mean_and_standard_deviation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-variable mean and standard deviation [D] over the observational
        rows (rows with no intervened cell), population standard deviation
        floored at MINIMUM_STANDARD_DEVIATION. This is the per-task
        standardisation of CITNP's training code (`_normalisation_stats`;
        equality tested in the CITNP integration tests)."""
        observational = ~self.intervened.any(-1)
        if not bool(observational.any()):
            raise ValueError("the context has no observational rows")
        rows = self.values[observational]
        mean = rows.mean(0)
        std = ((rows - mean) ** 2).mean(0).sqrt()
        return mean, std.clamp(min=MINIMUM_STANDARD_DEVIATION)

    def newest_regime_size(self) -> int | None:
        """Rows intervened on the node of the last row (None if the last row
        is observational or the context is empty)."""
        if self.num_rows == 0 or not bool(self.intervened[-1].any()):
            return None
        node = int(self.intervened[-1].nonzero()[0, 0])
        return int(self.intervened[:, node].sum())


def append_rows_to_context(
    context_values: torch.Tensor,
    context_intervened: torch.Tensor,
    new_rows: torch.Tensor,
    target_node: int | None | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append rows to contexts; the new rows are flagged as intervened on
    `target_node` and on nothing else (on nothing at all when None or < 0:
    an observational row). Rows go at the end: context rows are exchangeable.

    Args:
    ----------
    context_values : torch.Tensor
        [..., N, D] float.

    context_intervened : torch.Tensor
        [..., N, D] bool.

    new_rows : torch.Tensor
        [..., R, D] float, leading axes as the context's.

    target_node : int | None | torch.Tensor
        One node for all contexts, or a long tensor [...] with one node per
        context (-1 = observational).

    Returns:
    ----------
    (values [..., N + R, D], intervened [..., N + R, D]).
    """
    new_rows = torch.as_tensor(new_rows, dtype=context_values.dtype)
    D = context_values.shape[-1]
    flags = torch.zeros(new_rows.shape, dtype=torch.bool)
    if isinstance(target_node, torch.Tensor):
        node = target_node.to(torch.long)
        one_hot = torch.nn.functional.one_hot(node.clamp(min=0), D).bool()
        one_hot = one_hot & (node >= 0)[..., None]
        flags = flags | one_hot[..., None, :]
    elif target_node is not None:
        flags[..., int(target_node)] = True
    values = torch.cat([context_values, new_rows], dim=-2)
    intervened = torch.cat([context_intervened, flags], dim=-2)
    return values, intervened


def stack_contexts(
    contexts: Sequence[ExperimentContext],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contexts of equal size as ([B, N, D] values, [B, N, D] flags)."""
    if not contexts:
        raise ValueError("no contexts to stack")
    shape = contexts[0].values.shape
    if any(c.values.shape != shape for c in contexts):
        raise ValueError("stacked contexts must have equal shapes")
    return (
        torch.stack([c.values for c in contexts]),
        torch.stack([c.intervened for c in contexts]),
    )
