"""Linear probes on fixed feature matrices. Never calls a model.

`fit_linear_probe` answers "how well can a linear classifier recover the label
from these features?", estimated out of sample with folds split by group (one
group per context, so cells of one context never sit in both the fitting and
the scoring fold). The classifier is L2-regularised logistic regression fitted
by L-BFGS (limited-memory Broyden-Fletcher-Goldfarb-Shanno) on standardised
features, with classes weighted to equal total weight.

AUC = area under the ROC (receiver operating characteristic) curve: 0.5 is
chance, 1 is perfect separation. Balanced accuracy = mean of the true-positive
and true-negative rates at threshold 0.5 (also 0.5 at chance).
"""

import numpy as np
import torch
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean of true-positive and true-negative rates for scores > 0."""
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) > 0
    if labels.all() or (~labels).all():
        return float("nan")
    return float(0.5 * (pred[labels].mean() + (~pred[~labels]).mean()))


def _fit_logistic(
    X: torch.Tensor, y: torch.Tensor, l2: float, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced L2 logistic regression; returns (w [F], b [])."""
    w = torch.zeros(X.shape[1], dtype=X.dtype, requires_grad=True)
    b = torch.zeros((), dtype=X.dtype, requires_grad=True)
    pos = y.mean().clamp(1e-6, 1 - 1e-6)
    weight = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.LBFGS(
        [w, b], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=weight
            )
            + l2 * (w * w).sum()
        )
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 5,
    l2: float = 1e-3,
    max_iter: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Out-of-fold performance of a linear probe predicting y from X.

    X [n, F] features, y [n] binary labels, groups [n] group ids (folds never
    split a group). Returns auc and bal_acc (pooled over the out-of-fold
    scores), auc_se (standard error over folds), n and n_pos.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} groups, got {len(uniq)}"
        )
    rng = np.random.default_rng(seed)
    fold_of_group = dict(
        zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds)
    )
    fold = np.array([fold_of_group[g] for g in groups])

    scores = np.zeros(len(y))
    fold_aucs = []
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = torch.as_tensor((X[tr] - mu) / sd)
        Xte = torch.as_tensor((X[te] - mu) / sd)
        w, b = _fit_logistic(
            Xtr, torch.as_tensor(y[tr], dtype=Xtr.dtype), l2, max_iter
        )
        scores[te] = (Xte @ w + b).numpy()
        fold_aucs.append(roc_auc(scores[te], y[te]))
    fold_aucs = np.array([a for a in fold_aucs if np.isfinite(a)])
    return dict(
        auc=roc_auc(scores, y),
        auc_se=float(fold_aucs.std(ddof=1) / np.sqrt(len(fold_aucs)))
        if len(fold_aucs) > 1
        else float("nan"),
        bal_acc=balanced_accuracy(scores, y),
        n=int(len(y)),
        n_pos=int(y.sum()),
    )
