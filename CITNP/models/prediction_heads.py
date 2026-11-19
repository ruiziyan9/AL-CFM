"""Prediction heads of the mixed-context models, as mixins.

Every encoding family (`_MixedContextMixin`, `_RegimeAxisMixin`,
`_RegimeAxisFullMixin`) splits its forward pass in two steps:

    query_tokens = self._encode_query_tokens(...)   # [B, Nq, D, d_model]
    return self._predict_from_query_tokens(
        query_tokens, intervention_index, outcome_index
    )

The second step is the prediction head, swapped by the mixins here:

- `MarginalPredictionHeadMixin` (every marginal model): the univariate
  mixture p(X_Y | do(X_T = t), context) read from the outcome node's token.
- `JointPredictionHeadMixin` (the Joint* models): the joint mixture
  p(X_{-T} | do(X_T = t), context) over all non-intervened variables, read
  from the tokens of all variables (`JointMoGPredictor`,
  `JointMixtureGaussianLoss`).

`OutcomeFreeEmbeddingMixin` is the encoder side of the joint models: no
variable gets the outcome role.

Encoding contract of the joint head: an encoding supports it if (a) it
returns one token per variable for every query row, [B, Nq, D, d_model];
(b) query rows never exchange information with each other; (c) variable
roles come from `CausalInfEncoder.embed`, so the outcome role can be switched
off in one place. A new encoding gets the joint head by meeting the contract
and adding an outcome-free encoder class, a joint model class and one entry
of `JOINT_INTERVENTION_ENCODINGS` (CITNP/models/mixed_context_model.py).

This module must not import CITNP.models.mixed_context_model or the
regime-axis packages (they import it).
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from CITNP.models.causalinferencemodel import CausalInferenceModel
from CITNP.models.causaltransformercomponents import CausalInfEncoder
from CITNP.models.nncomponents import JointMoGPredictor
from CITNP.utils.loss import JointMixtureGaussianLoss
from CITNP.utils.outputs import JointModelOutput, ModelOutput
from CITNP.utils.utils import gather_by_node

PREDICTION_HEADS = ("marginal", "joint")
DEFAULT_PREDICTION_HEAD = "marginal"

# Defaults of the joint-head config keys (CausalInfModelConfig).
DEFAULT_JOINT_COVARIANCE_RANK = 8
DEFAULT_JOINT_MINIMUM_VARIANCE = 1.0e-4
DEFAULT_JOINT_HEAD_NODE_ATTENTION_LAYERS = 0
# Model-config keys that apply to the joint head only.
JOINT_CONFIG_KEYS = (
    "joint_covariance_rank",
    "joint_minimum_variance",
    "joint_head_node_attention_layers",
)


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_joint_head_settings(
    settings: Mapping, prediction_head: str, num_mixture_components
) -> dict:
    """Validate the joint-head keys of a model config, before the dataclass
    is built (its defaults cannot tell "not given" from "given").

    Args:
    -----
    - settings: the model settings (command line and YAML); a key that is
      absent or None counts as not given.
    - prediction_head: the resolved head of the model class, "marginal" or
      "joint".
    - num_mixture_components: K of the config.

    Returns:
    --------
    {key: value} of the joint keys that were given, to pass to
    CausalInfModelConfig (absent keys keep the dataclass defaults).

    Raises ValueError if a joint key is given while the head is marginal, if
    joint_covariance_rank is not an integer >= 1, joint_minimum_variance not
    a finite number > 0, joint_head_node_attention_layers not an integer
    >= 0, or (joint head) num_mixture_components < 1.
    """
    given = {
        k: settings[k] for k in JOINT_CONFIG_KEYS if settings.get(k) is not None
    }
    if prediction_head not in PREDICTION_HEADS:
        raise ValueError(
            f"Unknown prediction_head {prediction_head!r}; "
            f"choose from {list(PREDICTION_HEADS)}"
        )
    if prediction_head != "joint":
        if given:
            raise ValueError(
                f"{sorted(given)} only apply to the joint prediction head "
                "(prediction_head: joint); remove them for a marginal model"
            )
        return given
    rank = given.get("joint_covariance_rank", DEFAULT_JOINT_COVARIANCE_RANK)
    if not (_is_integer(rank) and rank >= 1):
        raise ValueError(
            f"joint_covariance_rank must be an integer >= 1 (got {rank!r})"
        )
    variance = given.get("joint_minimum_variance", DEFAULT_JOINT_MINIMUM_VARIANCE)
    if not (
        isinstance(variance, (int, float))
        and not isinstance(variance, bool)
        and math.isfinite(variance)
        and variance > 0
    ):
        raise ValueError(
            "joint_minimum_variance must be a finite number > 0 "
            f"(got {variance!r})"
        )
    layers = given.get(
        "joint_head_node_attention_layers",
        DEFAULT_JOINT_HEAD_NODE_ATTENTION_LAYERS,
    )
    if not (_is_integer(layers) and layers >= 0):
        raise ValueError(
            "joint_head_node_attention_layers must be an integer >= 0 "
            f"(got {layers!r})"
        )
    if not (_is_integer(num_mixture_components) and num_mixture_components >= 1):
        raise ValueError(
            "num_mixture_components must be >= 1 for the joint head "
            f"(got {num_mixture_components!r})"
        )
    return given


# The mixins call upstream model / encoder methods; give type checkers those
# bases only.
if TYPE_CHECKING:
    _ModelMixinBase = CausalInferenceModel
    _EncoderMixinBase = CausalInfEncoder
else:
    _ModelMixinBase = object
    _EncoderMixinBase = object


class MarginalPredictionHeadMixin(_ModelMixinBase):
    """The univariate head: gather the outcome node's token of every query
    row and run the model's predictor (Gaussian or mixture of Gaussians)."""

    prediction_head: str = "marginal"

    def _predict_from_query_tokens(
        self,
        query_tokens: Tensor,
        intervention_index: Tensor,
        outcome_index: Optional[Tensor],
    ) -> ModelOutput | JointModelOutput:
        """
        Args:
        -----
        - query_tokens: [B, Nq, D, d_model].
        - intervention_index: [B] long (T); unused here.
        - outcome_index: [B] long (Y); required.

        Returns:
        --------
        ModelOutput with pred_mean / pred_std (and weights for the mixture
        predictor), each [B, Nq, 1, ...].
        """
        if outcome_index is None:
            raise ValueError("the marginal prediction head needs outcome_index")
        # shape outcome_encoding: (batch_size, num_target, d_model)
        outcome_encoding = gather_by_node(query_tokens, outcome_index)
        if self.predictor.output_heads == ("mean", "std"):
            mean, std = self._predict(outcome_encoding)
            return ModelOutput(pred_mean=mean, pred_std=std)
        mean, std, weights = self._predict(outcome_encoding)
        return ModelOutput(pred_mean=mean, pred_std=std, weights=weights)


class JointPredictionHeadMixin(_ModelMixinBase):
    """The joint head: per query row, a K-component mixture of D-variate
    Gaussians (`JointMoGPredictor`) trained with the exact negative
    log-likelihood of the non-intervened variables
    (`JointMixtureGaussianLoss`).

    Must come first in the bases of a joint model class, before the marginal
    model it extends (whose encoder, embedding and target masking it keeps).
    Replaces `self.predictor`, `self.loss` and `self.eval_loss` after the
    marginal model is built. Reads the config keys
    num_mixture_components, joint_covariance_rank, joint_minimum_variance and
    joint_head_node_attention_layers.

    The model is a conditional density estimator for
    p(X_{-T} | do(X_T = t), context): one distribution per query row, each
    evaluated at the row's own do-value. Nothing enforces consistency between
    predictions for different interventions. Exactly one intervened variable
    per query group (`intervention_index` [B]).
    """

    prediction_head: str = "joint"

    def __init__(self, config):
        super().__init__(config)  # type: ignore[call-arg]
        self.predictor = JointMoGPredictor(
            d_model=self.d_model,
            depth=self.emb_depth,
            num_mixture_components=config.num_mixture_components,
            covariance_rank=getattr(
                config, "joint_covariance_rank", DEFAULT_JOINT_COVARIANCE_RANK
            ),
            minimum_variance=getattr(
                config, "joint_minimum_variance", DEFAULT_JOINT_MINIMUM_VARIANCE
            ),
            node_attention_layers=getattr(
                config,
                "joint_head_node_attention_layers",
                DEFAULT_JOINT_HEAD_NODE_ATTENTION_LAYERS,
            ),
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
        )
        self.loss = JointMixtureGaussianLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="mean",
        )
        self.eval_loss = JointMixtureGaussianLoss(
            mean_loss_across_samples=self.mean_loss_across_samples,
            reduce="mean",
        )

    def _predict_from_query_tokens(
        self,
        query_tokens: Tensor,
        intervention_index: Tensor,
        outcome_index: Optional[Tensor],
    ) -> ModelOutput | JointModelOutput:
        """
        Args:
        -----
        - query_tokens: [B, Nq, D, d_model], every variable of every row.
        - intervention_index: [B] long (T), one intervened variable per row.
        - outcome_index: ignored (the joint head has no outcome).

        Returns:
        --------
        JointModelOutput: component_mean [B, Nq, K, D], covariance_factor
        [B, Nq, K, D, R], covariance_diagonal [B, Nq, K, D], mixture_logits
        [B, Nq, K], intervened_mask [B, D] (one-hot of intervention_index).
        """
        mean, factor, diagonal, logits = self.predictor(query_tokens)
        num_nodes = query_tokens.shape[-2]
        intervened_mask = F.one_hot(intervention_index, num_nodes).bool()
        return JointModelOutput(
            component_mean=mean,
            covariance_factor=factor,
            covariance_diagonal=diagonal,
            mixture_logits=logits,
            intervened_mask=intervened_mask,
        )

    def calculate_loss(
        self,
        model_output: ModelOutput | JointModelOutput,
        target: Tensor,
        outcome_index: Optional[Tensor] = None,
        test: bool = False,
    ) -> Tensor:
        """Joint negative log-likelihood of the non-intervened variables.

        Args:
        -----
        - model_output: JointModelOutput of this model.
        - target: [B, Nq, D, 1] (or [B, Nq, D]), standardised full query rows.
        - outcome_index: ignored.
        - test: per-element losses [B] (evaluation only), as
          CausalInferenceModel.calculate_loss.
        """
        if not isinstance(model_output, JointModelOutput):
            raise TypeError("the joint head expects a JointModelOutput")
        if target.ndim == 4:
            target = target.squeeze(-1)  # [B, Nq, D]
        # self.loss / self.eval_loss are typed BaseLoss upstream; this mixin
        # sets both to JointMixtureGaussianLoss
        if self.training:
            loss = cast(JointMixtureGaussianLoss, self.loss)
            return loss.calculate_loss(model_output, target)
        eval_loss = cast(JointMixtureGaussianLoss, self.eval_loss)
        return eval_loss.calculate_loss(model_output, target, test=test)


class OutcomeFreeEmbeddingMixin(_EncoderMixinBase):
    """`CausalInfEncoder.embed` without the outcome role: every
    non-intervened node, in context and query rows, is embedded with the
    variable-role network; the intervened node with the intervention-role
    network. The outcome networks (`context_outcome`, `target_outcome`) stay
    in the module, unused, so encoder parameter names match marginal
    checkpoints (the computation does not).

    Must come first in the bases of an outcome-free encoder class.
    """

    def embed(self, data, outcome_indices, intervention_indices, context=True):
        """
        Args:
        -----
        - data: [batch_size, num_samples, num_nodes, 1].
        - outcome_indices: ignored ([batch_size] or None).
        - intervention_indices: [batch_size] long.

        Returns:
        --------
        embedding: [batch_size, num_samples, num_nodes, d_model].
        """
        if context:
            var_emb = self.context_variable(data)
            int_emb = self.context_intervention(data)
        else:
            var_emb = self.target_variable(data)
            int_emb = self.target_intervention(data)
        batch_idx = torch.arange(data.shape[0], device=data.device)
        mask_int = torch.zeros_like(var_emb)
        mask_int[batch_idx[:, None], :, intervention_indices[:, None], :] = 1.0
        return var_emb * (1 - mask_int) + int_emb * mask_int
