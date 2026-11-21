"""Active learning on a joint model: choose the next intervention (node and
value) that is most informative about a target query
p(X_Y | do(X_T = psi)), run it, add it to the context, repeat.

See CITNP/active_learning/README.md. Models are used only through
`CITNP.evaluation.model_api` (`ModelApiPredictiveModel`).
"""

from CITNP.active_learning.config import (
    ActiveLearningConfig,
    CandidateExperiment,
    candidate_target_nodes,
)
from CITNP.active_learning.design_optimisation import (
    BayesianOptimisationValueOptimiser,
    CandidateScore,
    GridValueOptimiser,
    InterventionValueOptimiser,
    ValueDomain,
    ValueEvaluation,
    ValueOptimisationResult,
    choice_is_clear,
)
from CITNP.active_learning.environment import (
    ExperimentEnvironment,
    LinearSCMEnvironment,
)
from CITNP.active_learning.fantasy import (
    ExperimentContext,
    append_rows_to_context,
    sample_fantasy_rows,
)
from CITNP.active_learning.information_gain import (
    InformationGainEstimate,
    estimate_expected_information_gain,
)
from CITNP.active_learning.loop import (
    ActiveLearningGenerators,
    ActiveLearningStep,
    run_active_learning,
    select_next_experiment,
)
from CITNP.active_learning.predictive_model import (
    ModelApiPredictiveModel,
    PredictiveModel,
    QueryMixture,
)
from CITNP.active_learning.training_support import TrainingSupport

__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningGenerators",
    "ActiveLearningStep",
    "BayesianOptimisationValueOptimiser",
    "CandidateExperiment",
    "CandidateScore",
    "ExperimentContext",
    "ExperimentEnvironment",
    "GridValueOptimiser",
    "InformationGainEstimate",
    "InterventionValueOptimiser",
    "LinearSCMEnvironment",
    "ModelApiPredictiveModel",
    "PredictiveModel",
    "QueryMixture",
    "TrainingSupport",
    "ValueDomain",
    "ValueEvaluation",
    "ValueOptimisationResult",
    "append_rows_to_context",
    "candidate_target_nodes",
    "choice_is_clear",
    "estimate_expected_information_gain",
    "run_active_learning",
    "sample_fantasy_rows",
    "select_next_experiment",
]
