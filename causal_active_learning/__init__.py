"""Causal active learning: choose the next acquisition (R rows under an
intervention on a node at a value, or R observational rows) that is most
informative about a target query p(X_Y | do(X_T = psi)), run it, add it to the
context, repeat.

The core knows nothing about any particular model: a model implements
`CausalPredictiveModel` (protocols.py). The CITNP adapter lives in
`causal_active_learning.integrations.citnp`. See README.md.
"""

from causal_active_learning.acquisitions import (
    Acquisition,
    CandidateSet,
    ConstrainedEnvironment,
    InterventionalQuery,
    NonIntervenableNodeError,
    build_candidate_set,
)
from causal_active_learning.config import ActiveLearningConfig
from causal_active_learning.context import ExperimentalContext
from causal_active_learning.distributions import UnivariateGaussianMixture
from causal_active_learning.information_gain import (
    InformationGainEstimate,
    estimate_expected_information_gain,
)
from causal_active_learning.loop import (
    ActiveLearningGenerators,
    ActiveLearningStep,
    run_active_learning,
)
from causal_active_learning.protocols import (
    CausalPredictiveModel,
    ExperimentEnvironment,
)
from causal_active_learning.selection import (
    AcquisitionScore,
    choice_is_clear,
    select_next_acquisition,
)
from causal_active_learning.support_check import TrainingSupport

__all__ = [
    "Acquisition",
    "AcquisitionScore",
    "ActiveLearningConfig",
    "ActiveLearningGenerators",
    "ActiveLearningStep",
    "CandidateSet",
    "CausalPredictiveModel",
    "ConstrainedEnvironment",
    "ExperimentEnvironment",
    "ExperimentalContext",
    "InformationGainEstimate",
    "InterventionalQuery",
    "NonIntervenableNodeError",
    "TrainingSupport",
    "UnivariateGaussianMixture",
    "build_candidate_set",
    "choice_is_clear",
    "estimate_expected_information_gain",
    "run_active_learning",
    "select_next_acquisition",
]
