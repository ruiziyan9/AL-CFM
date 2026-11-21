"""Fantasy noise and fantasy contexts.

A fantasy of an acquisition is a set of R rows drawn from the model's
predictive distribution for that acquisition (the BO convention, as in
BoTorch's `fantasize`); the fantasy context is the current context with one
fantasy appended. The model draws the rows (`PredictiveModel.sample_fantasies`)
from standard noise that the core supplies, so the core decides whether noise
is fresh for every evaluation or shared.
"""

from collections.abc import Sequence
from typing import Any

import torch

from causal_active_learning.acquisitions import CandidateExperiment
from causal_active_learning.context import ExperimentContext
from causal_active_learning.protocols import PredictiveModel


def draw_fantasy_noise(
    model: PredictiveModel,
    nodes: Sequence[int | None],
    num_fantasies: int,
    num_variables: int,
    num_rows: int,
    generator: torch.Generator,
) -> dict[int | None, Any]:
    """One independent noise draw per node (None: observational), drawn in
    the given order from `generator`."""
    return {
        node: model.draw_fantasy_noise(
            num_variables, num_fantasies, num_rows, generator
        )
        for node in nodes
    }


def fantasy_contexts(
    context: ExperimentContext,
    candidates: Sequence[CandidateExperiment],
    fantasies: torch.Tensor,
) -> list[ExperimentContext]:
    """The C x S fantasy contexts (candidate-major) of fantasies [C, S, R, D]:
    the context with all R rows of fantasy s of candidate c appended, flagged
    as intervened on the candidate's node."""
    out = []
    for c, candidate in enumerate(candidates):
        for s in range(fantasies.shape[1]):
            out.append(context.append(fantasies[c, s], candidate.target_node))
    return out
