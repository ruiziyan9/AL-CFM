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

`fit_classification_probe` is the multiclass version (softmax regression,
same folds, standardisation and class weighting), reporting accuracy and
balanced accuracy (mean per-class recall; 1 / number of classes at chance).
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


def assign_group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index [n] of every sample such that all samples of one group
    share a fold: the distinct groups, shuffled with `seed`, are dealt to the
    folds in turn. Raises ValueError for fewer groups than folds."""
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
    return np.array([fold_of_group[g] for g in groups])


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
    fold = assign_group_folds(groups, n_folds, seed)

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


def _fit_softmax(
    X: torch.Tensor, y: torch.Tensor, num_classes: int, l2: float, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced L2 softmax regression; returns (W [F, C], b [C]).

    Every class present in y gets the same total weight (classes absent from
    y get none), so the minority class is not ignored."""
    W = torch.zeros(X.shape[1], num_classes, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(num_classes, dtype=X.dtype, requires_grad=True)
    counts = torch.bincount(y, minlength=num_classes).to(X.dtype)
    num_present = (counts > 0).sum()
    sample_weight = (1.0 / (num_present * counts.clamp_min(1)))[y]
    opt = torch.optim.LBFGS(
        [W, b], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = (
            sample_weight
            * torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none")
        ).sum() + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def multiclass_balanced_accuracy(
    predictions: np.ndarray, labels: np.ndarray
) -> float:
    """Mean over the classes present in `labels` of the fraction of that
    class predicted correctly (recall)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    recalls = [
        float((predictions[labels == c] == c).mean()) for c in np.unique(labels)
    ]
    return float(np.mean(recalls))


def fit_classification_probe(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 5,
    l2: float = 1e-3,
    max_iter: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Out-of-fold performance of a linear (softmax) probe predicting integer
    class labels from X; binary or multiclass.

    X [n, F] features, labels [n] ints in 0..C - 1, groups [n] group ids
    (folds never split a group, see `assign_group_folds`). Features are
    standardised with the fitting fold's mean and std. Metrics are pooled
    over the out-of-fold predictions:
        accuracy, balanced_accuracy (mean per-class recall),
        auc (binary only, from the logit difference; NaN otherwise),
        majority_class_rate (accuracy of always predicting the most
        frequent class), n, num_classes (C), num_groups.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    fold = assign_group_folds(groups, n_folds, seed)
    logits = np.zeros((len(labels), num_classes))
    for k in range(n_folds):
        fit_rows, score_rows = fold != k, fold == k
        mean, std = X[fit_rows].mean(0), X[fit_rows].std(0) + 1e-8
        X_fit = torch.as_tensor((X[fit_rows] - mean) / std)
        X_score = torch.as_tensor((X[score_rows] - mean) / std)
        W, b = _fit_softmax(
            X_fit, torch.as_tensor(labels[fit_rows]), num_classes, l2, max_iter
        )
        logits[score_rows] = (X_score @ W + b).numpy()
    predictions = logits.argmax(1)
    auc = float("nan")
    if num_classes == 2:
        auc = roc_auc(logits[:, 1] - logits[:, 0], labels == 1)
    return dict(
        accuracy=float((predictions == labels).mean()),
        balanced_accuracy=multiclass_balanced_accuracy(predictions, labels),
        auc=auc,
        majority_class_rate=float(np.bincount(labels).max() / len(labels)),
        n=int(len(labels)),
        num_classes=num_classes,
        num_groups=int(len(np.unique(groups))),
    )


def shuffle_labels(labels: np.ndarray, seed: int | list[int]) -> np.ndarray:
    """A random permutation of `labels` (the shuffled-label control): same
    class frequencies, no relation to the features."""
    return np.random.default_rng(seed).permutation(np.asarray(labels))
