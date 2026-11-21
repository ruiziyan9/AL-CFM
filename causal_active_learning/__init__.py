"""Causal active learning: choose the next acquisition (an intervention on a
node at a value, or observational rows) that is most informative about a
target query p(X_Y | do(X_T = psi)), run it, add it to the context, repeat.

The core knows nothing about any particular model: a model implements
`CausalPredictiveModel` (protocols.py). The CITNP adapter lives in
`causal_active_learning.integrations.citnp`. See README.md.
"""

from causal_active_learning.acquisitions import (
    Acquisition,
    candidate_target_nodes,
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
    ExperimentEnvironment,
    CausalPredictiveModel,
)
from causal_active_learning.selection import (
    AcquisitionScore,
    choice_is_clear,
    select_next_acquisition,
)
from causal_active_learning.support_check import TrainingSupport

__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningGenerators",
    "ActiveLearningStep",
    "Acquisition",
    "AcquisitionScore",
    "ExperimentalContext",
    "ExperimentEnvironment",
    "InformationGainEstimate",
    "CausalPredictiveModel",
    "UnivariateGaussianMixture",
    "TrainingSupport",
    "candidate_target_nodes",
    "choice_is_clear",
    "estimate_expected_information_gain",
    "run_active_learning",
    "select_next_acquisition",
]
