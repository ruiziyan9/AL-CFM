import math
from abc import ABC, abstractmethod
from typing import Optional

import einops
import torch

from CITNP.utils.outputs import JointModelOutput, ModelOutput


class BaseLoss(ABC):
    def __init__(self, mean_loss_across_samples: bool, reduce: str):
        self.mean_loss_across_samples = mean_loss_across_samples
        assert reduce in ["mean", "sum"]
        self.reduce = reduce

    @abstractmethod
    def calculate_loss(
        self,
        model_output: ModelOutput,
        trgt_outcome: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


class CNPLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_loss(
        self,
        model_output: ModelOutput,
        trgt_outcome: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std

        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)
        log_prob = distribution.log_prob(trgt_outcome)

        if self.mean_loss_across_samples:
            log_prob_all_samples = log_prob.mean(1).sum(-1)
        else:
            log_prob_all_samples = log_prob.sum(1).sum(-1)

        if self.reduce == "mean":
            log_prob_allbatch = log_prob_all_samples.mean(0)
        elif self.reduce == "sum":
            log_prob_allbatch = log_prob_all_samples.sum(0)

        neg_log_prob_allbatch = -log_prob_allbatch
        if not test:
            return neg_log_prob_allbatch
        else:
            return -log_prob_all_samples


class MixtureGaussianLoss(BaseLoss):
    def __init__(
        self,
        num_mixture_components: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixture_components = num_mixture_components

    def calculate_loss(
        self,
        model_output: ModelOutput,
        trgt_outcome: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std
        weights = model_output.weights
        _, _, num_outcome_nodes, num_components = pred_mean.shape
        assert num_outcome_nodes == trgt_outcome.shape[-1], (
            "Mismatch between number of outcome nodes in model output and target outcome"
        )

        # Expand target to match mixture dimensions
        target_expanded = einops.repeat(
            trgt_outcome, "b nt d -> b nt d K", K=num_components
        )

        # Define the Gaussian distributions for each component
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)

        # Compute log probabilities for all components
        log_prob = distribution.log_prob(
            target_expanded
        )  # Shape: (batch_size, num_samples, d, K)

        assert weights is not None, "weights must not be None"
        # Weighted log probabilities with mixture weights
        weighted_log_prob = torch.logsumexp(
            log_prob + torch.log(weights + 1e-12), dim=-1
        )  # Shape: (batch_size, num_samples, d)

        # Reduce over the samples
        if self.mean_loss_across_samples:
            log_prob_all_samples = weighted_log_prob.mean(1).sum(-1)
        else:
            log_prob_all_samples = weighted_log_prob.sum(1).sum(-1)

        if self.reduce == "mean":
            log_prob_allbatch = log_prob_all_samples.mean(0)
        elif self.reduce == "sum":
            log_prob_allbatch = log_prob_all_samples.sum(0)

        neg_log_prob_allbatch = -log_prob_allbatch
        if not test:
            return neg_log_prob_allbatch
        else:
            return -log_prob_all_samples


class NLLLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_loss(
        self,
        model_output: ModelOutput,
        trgt_outcome: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        pred_mean = model_output.pred_mean
        pred_std = model_output.pred_std

        num_z_samples = pred_mean.size(0)
        distribution = torch.distributions.Normal(pred_mean, pred_std + 1e-12)

        likelihood = distribution.log_prob(trgt_outcome)
        logL = torch.log(
            torch.tensor(
                num_z_samples,
                device=likelihood.device,
                dtype=likelihood.dtype,
            )
        )
        likelihood_z_meaned = torch.logsumexp(likelihood, dim=0) - logL

        if self.mean_loss_across_samples:
            likelihood_allsample = likelihood_z_meaned.mean(1).sum(-1)
        else:
            likelihood_allsample = likelihood_z_meaned.sum(1).sum(-1)

        if self.reduce == "mean":
            likelihood_allbatch = likelihood_allsample.mean(0)
        elif self.reduce == "sum":
            likelihood_allbatch = likelihood_allsample.sum(0)

        neglikelihood_allbatch = -likelihood_allbatch
        if not test:
            return neglikelihood_allbatch
        else:
            return -likelihood_allsample


LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def pad_intervened_variables(
    component_mean: torch.Tensor,
    covariance_factor: torch.Tensor,
    covariance_diagonal: torch.Tensor,
    intervened_mask: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn every intervened variable into a dummy coordinate (padding).

    For each intervened variable j: factor row -> 0, diagonal -> 1, mean ->
    the target value x_j. The covariance of every component is then
    block-diagonal: the block of the non-intervened variables is exactly
    F F^T + diag(v) restricted to them, the intervened block is the identity
    with no covariance to the rest, and every intervened residual is 0. Each
    dummy coordinate therefore contributes log N(x_j; x_j, 1) = -0.5 log(2 pi)
    to the log-density, identically in every component; see
    `JointMixtureGaussianLoss`. The head's outputs for intervened variables
    are never used. Uses torch.where (no in-place writes), so autograd is
    untouched and their gradients are exactly zero.

    Args:
    -----
    - component_mean: [B, Nq, K, D].
    - covariance_factor: [B, Nq, K, D, R].
    - covariance_diagonal: [B, Nq, K, D].
    - intervened_mask: [B, D] bool, True = set by do(); any number of True
      entries per row, different across rows.
    - target: [B, Nq, D] (or with extra leading sample dimensions that
      broadcast against [B, Nq, D] after the K axis is added, see below).

    Returns:
    --------
    (component_mean, covariance_factor, covariance_diagonal), padded, with the
    shapes above (the mean broadcast to the target's leading shape).
    """
    m = intervened_mask[:, None, None, :]  # [B, 1, 1, D]
    mean = torch.where(m, target.unsqueeze(-2), component_mean)
    factor = torch.where(
        m[..., None], torch.zeros_like(covariance_factor), covariance_factor
    )
    diagonal = torch.where(
        m, torch.ones_like(covariance_diagonal), covariance_diagonal
    )
    return mean, factor, diagonal


def joint_mixture_log_density(
    model_output: JointModelOutput, target: torch.Tensor
) -> torch.Tensor:
    """log p(target of the non-intervened variables) under the joint mixture,
    in float64, per query row.

    It is a density over D minus the number of intervened variables of the
    row, not over D variables. A component whose Cholesky factorisation fails
    gives NaN for that element (no exception), so a training loss built from
    it is non-finite and the step is skipped.

    Args:
    -----
    - model_output: JointModelOutput with [B, Nq, ...] parameters.
    - target: [B, Nq, D], same scale as the parameters.

    Returns:
    --------
    log density [B, Nq], float64.
    """
    # float64 is required, not optional: with the diagonal near its floor a
    # float32 likelihood loses whole nats, while float64 with a dense
    # Cholesky stays within 1e-5. The matrices are D x D, so it is cheap.
    target = target.double()
    mean, factor, diagonal = pad_intervened_variables(
        model_output.component_mean.double(),
        model_output.covariance_factor.double(),
        model_output.covariance_diagonal.double(),
        model_output.intervened_mask,
        target,
    )
    # Dense covariance [B, Nq, K, D, D] and its Cholesky factor. Not
    # LowRankMultivariateNormal: its Woodbury form subtracts two large numbers
    # when the diagonal is small, and it only pays off for D >> rank.
    covariance = factor @ factor.transpose(-1, -2) + torch.diag_embed(diagonal)
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    failed = info > 0  # [B, Nq, K]
    # a failed factor holds garbage; give MultivariateNormal a valid one and
    # mark the element as NaN below
    eye = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    cholesky = torch.where(failed[..., None, None], eye, cholesky)
    mixture = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(
            logits=model_output.mixture_logits.double(), validate_args=False
        ),
        torch.distributions.MultivariateNormal(
            mean, scale_tril=cholesky, validate_args=False
        ),
        validate_args=False,
    )
    log_prob = mixture.log_prob(target)  # [B, Nq]
    log_prob = torch.where(
        failed.any(-1), torch.full_like(log_prob, float("nan")), log_prob
    )
    # Remove the padding constant. Each intervened variable is a dummy
    # coordinate contributing log N(x; x, 1) = -0.5 log(2 pi) in every
    # mixture component, so the factor leaves the mixture sum exactly and
    # adding it back gives the log-density of the non-intervened variables.
    # Rows may have different numbers of intervened variables: padding keeps
    # every covariance D x D, so all rows share one batched Cholesky call;
    # gathering the non-intervened variables instead would give matrices of
    # different sizes, which cannot be stacked into one tensor.
    num_intervened = model_output.intervened_mask.sum(-1).to(log_prob.dtype)
    return log_prob + LOG_SQRT_TWO_PI * num_intervened[:, None]


class JointMixtureGaussianLoss(BaseLoss):
    """Exact negative log-likelihood of the joint mixture over the
    non-intervened variables of each query row (`JointMoGPredictor`).

    Computed in float64 with a dense Cholesky factorisation; intervened
    variables are removed by padding (`pad_intervened_variables`), so the
    intervened mask may differ between rows, in position and in count, and
    everything stays one batched computation.

    Reduction exactly as `MixtureGaussianLoss`: mean (mean_loss_across_samples)
    or sum over the Nq query rows, then mean or sum (`reduce`) over B;
    test=True returns the per-element vector [B].

    The loss is a density over the non-intervened variables only (D - 1 of
    them with one intervention), so its value is NOT comparable with the
    marginal (one-variable) loss.

    Weighting across rows: each row's term is a log-density over its own
    number of non-intervened variables, so a row predicting more variables
    has a larger term, roughly in proportion. This comes from the joint
    likelihood, not from the padding, and does not change the optimum (each
    row's term is a proper scoring rule, minimised by the true conditional
    joint, whatever positive weight the row gets); it only changes how a
    finite model trades accuracy between rows of different sizes. With one
    intervention per row (the contract of this model) the weighting is
    uniform. When rows with different numbers of intervened variables are
    trained together, dividing each row's term by its number of
    non-intervened variables is the alternative to consider; the plain sum is
    implemented here.
    """

    def calculate_loss(  # type: ignore[override]  # joint output
        self,
        model_output: JointModelOutput,
        target: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        """
        Args:
        -----
        - model_output: JointModelOutput, parameters [B, Nq, ...].
        - target: [B, Nq, D] on the standardised scale.
        - test: return the per-element losses [B] instead of the reduction.

        Returns:
        --------
        scalar loss (float64), or [B] with test=True.
        """
        log_prob = joint_mixture_log_density(model_output, target)  # [B, Nq]
        if self.mean_loss_across_samples:
            log_prob_all_samples = log_prob.mean(1)
        else:
            log_prob_all_samples = log_prob.sum(1)

        if self.reduce == "mean":
            log_prob_allbatch = log_prob_all_samples.mean(0)
        else:
            log_prob_allbatch = log_prob_all_samples.sum(0)

        if not test:
            return -log_prob_allbatch
        return -log_prob_all_samples
