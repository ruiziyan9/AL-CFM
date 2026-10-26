"""Single source of truth for the per-node linear-Gaussian (NIG) prior.

Per node j with n_parents parents (x_j = w_j^T x_pa + e_j, e_j ~ N(0, s2_j)):

    s2_j ~ IG(node_shape(n_parents), ig_scale)
    w_j | s2_j ~ N(sign * weight_mean, eta * s2_j * I)   (sign = +-1 per edge)

IG = Inverse-Gamma (shape, scale). The generator (`functions_generator.py`) and
the exact oracle (`evaluation/oracle.py`) both read this dataclass, so they
cannot drift apart.

Families:
    legacy  roots IG(2.5, 0.5), every non-root IG(3.0, 0.5) whatever its number
            of parents (the original `linearnonidentifiable`).
    IW      one inverse-Wishart prior on the whole covariance, Sigma ~
            IW(nu, 2 beta I) with nu = 2 alpha0 + D - 1, means known to be zero.
            For any DAG it induces s2_j ~ IG(alpha0 + |pa_j| / 2, beta) and
            w_j | s2_j ~ N(0, eta s2_j I) with eta = 1 / (2 beta)
            (parameter modularity), which makes the prior score-equivalent.
    NIG     any other per-node setting (`linear_nig`).

"NIG" names the per-node form that all families share (so the oracle is closed
form for all of them); "IW" names the joint prior the IW family comes from.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Any

# Registry keys of the linear-Gaussian families.
LEGACY_KEYS = ("linearnonidentifiable", "linear_nig_legacy")
LINEAR_GAUSSIAN_KEYS = (*LEGACY_KEYS, "linear_nig", "linear_iw")

_LEGACY_ALPHA_ROOT = 2.5
_LEGACY_ALPHA_NODE = 3.0
_LEGACY_SCALE = 0.5

_ALIAS_WARNED = False

ALIAS_WARNING = (
    "linearnonidentifiable is the legacy NIG prior; it is score-equivalent "
    "only for graphs whose nodes have <= 1 parent. Use linear_nig_legacy, "
    "linear_nig or linear_iw."
)


def is_linear_gaussian(name: str) -> bool:
    """True for every registry key whose mechanisms are linear-Gaussian with a
    per-node NIG prior (`linearnonidentifiable`, `linear_nig_legacy`,
    `linear_nig`, `linear_iw`)."""
    return name in LINEAR_GAUSSIAN_KEYS


def is_legacy_key(name: str) -> bool:
    """True for the keys that use the original `LinearNIGLegacy` mechanism."""
    return name in LEGACY_KEYS


def warn_legacy_alias_once() -> None:
    """Emit the `linearnonidentifiable` DeprecationWarning once per process."""
    global _ALIAS_WARNED
    if not _ALIAS_WARNED:
        _ALIAS_WARNED = True
        warnings.warn(ALIAS_WARNING, DeprecationWarning, stacklevel=3)


@dataclass(frozen=True)
class LinearGaussianPrior:
    """Per-node NIG prior of a linear-Gaussian SCM.

    Args:
    ----------
    root_shape : float
        IG shape of a node with no parents (alpha0 under IW).

    shape_per_parent : float
        IG shape added per parent (0.5 under IW).

    nonroot_shape : float | None
        If set, a constant IG shape for every non-root, whatever its number of
        parents (the legacy setting); `shape_per_parent` is then unused.

    ig_scale : float
        IG scale (beta).

    eta : float
        Weight-prior scale: w | s2 ~ N(+-weight_mean, eta * s2 * I).

    weight_mean : float
        Signed weight mean m >= 0 (0 = zero-mean prior).
    """

    root_shape: float
    shape_per_parent: float
    nonroot_shape: float | None
    ig_scale: float
    eta: float
    weight_mean: float

    def __post_init__(self):
        """Validate the hyperparameters."""
        if not self.root_shape > 0 or not self.shape_per_parent > 0:
            raise ValueError("IG shapes must be > 0")
        if self.nonroot_shape is not None and not self.nonroot_shape > 0:
            raise ValueError("nonroot_shape must be > 0")
        if not self.ig_scale > 0:
            raise ValueError("ig_scale must be > 0")
        if not self.eta > 0:
            raise ValueError("eta must be > 0")
        if not self.weight_mean >= 0:
            raise ValueError("weight_mean must be >= 0")

    def node_shape(self, n_parents: int) -> float:
        """IG shape of the noise variance of a node with `n_parents` parents."""
        if n_parents == 0:
            return self.root_shape
        if self.nonroot_shape is not None:
            return self.nonroot_shape
        return self.root_shape + self.shape_per_parent * n_parents

    @property
    def is_iw_compatible(self) -> bool:
        """True iff the per-node prior is the one an inverse-Wishart prior on
        the whole covariance induces: shape alpha0 + |pa| / 2, zero weight mean
        and eta = 1 / (2 * ig_scale). IW compatibility implies score
        equivalence: Markov-equivalent DAGs have identical marginal
        likelihoods on observational data."""
        return (
            self.nonroot_shape is None
            and math.isclose(self.shape_per_parent, 0.5)
            and self.weight_mean == 0
            and math.isclose(self.eta, 1.0 / (2.0 * self.ig_scale))
        )

    @classmethod
    def legacy(cls, eta: float = 1.0, weight_mean: float = 0.0):
        """The original prior: roots IG(2.5, 0.5), non-roots IG(3.0, 0.5)."""
        return cls(
            root_shape=_LEGACY_ALPHA_ROOT,
            shape_per_parent=0.5,  # unused: nonroot_shape is set
            nonroot_shape=_LEGACY_ALPHA_NODE,
            ig_scale=_LEGACY_SCALE,
            eta=float(eta),
            weight_mean=float(weight_mean),
        )

    @classmethod
    def iw(cls, alpha0: float = 2.5, ig_scale: float = 0.5):
        """The prior induced by Sigma ~ IW(2 alpha0 + D - 1, 2 ig_scale I)."""
        return cls(
            root_shape=float(alpha0),
            shape_per_parent=0.5,
            nonroot_shape=None,
            ig_scale=float(ig_scale),
            eta=1.0 / (2.0 * ig_scale),
            weight_mean=0.0,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "LinearGaussianPrior":
        """Prior of a `MixedRegimeConfig` (non-linear-Gaussian families fall back to
        `legacy(weight_scale, weight_mean)`, a reference only).

        Legacy keys: `legacy(weight_scale, weight_mean)`. `linear_iw`:
        `iw(alpha0 or 2.5)`, which requires weight_scale == 1 and
        weight_mean == 0. `linear_nig`: alpha0 (default 2.5) is the root
        shape, 0.5 per parent, and weight_scale / weight_mean pass through.
        """
        name = cfg.function_generator
        eta = float(getattr(cfg, "weight_scale", 1.0))
        m = float(getattr(cfg, "weight_mean", 0.0))
        alpha0 = getattr(cfg, "alpha0", None)
        if name in LEGACY_KEYS:
            return cls.legacy(eta=eta, weight_mean=m)
        if name == "linear_iw":
            if eta != 1.0 or m != 0.0:
                raise ValueError(
                    "linear_iw fixes eta = 1 / (2 * 0.5) = 1 and weight_mean = 0; "
                    "use linear_nig for other weight priors"
                )
            return cls.iw(2.5 if alpha0 is None else alpha0)
        if name == "linear_nig":
            return replace(
                cls.iw(2.5 if alpha0 is None else alpha0), eta=eta, weight_mean=m
            )
        # Other priors: the oracle is only a reference, scored with the legacy NIG.
        return cls.legacy(eta=eta, weight_mean=m)
