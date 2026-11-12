"""Scores and baselines shared by the collection.

Every exam reports the model next to a classical method and a dumb baseline (guess the dev
median, or the majority class). These helpers make the dumb ones one line each.
"""
from __future__ import annotations

import numpy as np


# ---- errors -----------------------------------------------------------------------------------
def median_abs_error(pred, truth) -> float:
    return float(np.nanmedian(np.abs(np.asarray(pred, float) - np.asarray(truth, float))))


def median_rel_error(pred, truth) -> float:
    """Median |pred/truth − 1| (as a fraction)."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.nanmedian(np.abs(p / t - 1.0)))


def within_factor(pred, truth, factor: float = 2.0) -> float:
    """Fraction of predictions within a multiplicative factor of the truth (QuakeNet depth)."""
    r = np.asarray(pred, float) / np.asarray(truth, float)
    return float(np.mean((r <= factor) & (r >= 1.0 / factor)))


def skill(err_model, err_reference, floor: float | None = None) -> float:
    """Skill score ``1 − err_model / err_reference`` (1 = perfect, 0 = no better than the reference)."""
    s = 1.0 - float(err_model) / float(err_reference)
    return max(floor, s) if floor is not None else s


# ---- classification --------------------------------------------------------------------------
def confusion(pred, truth, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), int)
    for p, t in zip(pred, truth):
        m[idx[t], idx[p]] += 1
    return m


def balanced_accuracy(pred, truth, classes=None) -> float:
    """Mean per-class recall (BearingNet's headline)."""
    classes = list(classes or sorted(set(truth)))
    m = confusion(pred, truth, classes)
    rec = [m[i, i] / m[i].sum() for i in range(len(classes)) if m[i].sum()]
    return float(np.mean(rec))


def macro_f1(pred, truth, classes=None) -> float:
    """Unweighted mean F1 over classes (BeatNet's headline)."""
    classes = list(classes or sorted(set(truth) | set(pred)))
    m = confusion(pred, truth, classes)
    f1 = []
    for i in range(len(classes)):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i].sum() - m[i, i]
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def roc_auc(score, label) -> float:
    """Area under the ROC curve via the rank-sum (Mann–Whitney) statistic; ties count half."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# ---- dumb baselines --------------------------------------------------------------------------
def median_guess(dev_values):
    """Always guess the dev median. Fit on dev only."""
    m = float(np.nanmedian(np.asarray(dev_values, float)))
    return lambda n: np.full(n, m)


def majority_guess(dev_labels):
    """Always guess the most common dev class."""
    vals, counts = np.unique(np.asarray(dev_labels), return_counts=True)
    c = vals[counts.argmax()]
    return lambda n: np.full(n, c)
