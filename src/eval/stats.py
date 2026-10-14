"""Uncertainty and significance for the comparison table.

A test set of a few hundred recordings gives macro-F1 values that move by a
few points from one sample of files to the next, so a difference between
two models is only reported together with how sure we are of it:

  * bootstrap_ci: resample test files with replacement and take the 2.5 /
    97.5 percentiles of accuracy and macro-F1 (Efron & Tibshirani 1993).
  * paired_bootstrap: the same resampled files are scored for both models,
    so the spread of the difference excludes the variation both models share
    (which files happen to be hard). The two-sided p-value is twice the
    share of resampled differences on the smaller side of zero, so p < 0.05
    goes together with a 95% interval of the difference that excludes zero.
  * mcnemar: counts the files only one of the two models gets right; under
    "equally good" those disagreements split 50 / 50 (McNemar 1947). Exact
    binomial below 25 disagreements, chi-square with continuity correction
    above.
  * aggregate_seeds: mean and sample sd over training seeds, the variation
    that comes from initialisation and data order rather than the test set.

Macro-F1 is averaged over the species present in the labels being scored.
On the full test set, where every species occurs, this is exactly the
number from src/eval/metrics.py (labels=range(n_classes), zero_division=0).
Averaging over all n_classes inside a resample would give F1 = 0 to every
species the resample happened to leave out, and with one or two test files
per rare species that biases the interval downwards. The same holds for any
subset of the test files, such as the recordist-unseen ones; there it equals
metrics.scores(..., present_only=True), which src/train.py uses for them.

Prediction files are written by src/train.py as
results/predictions/<run>_<split>.npz with arrays files, labels, probs
(file-level mean of chunk softmax) and classes.
"""
from pathlib import Path

import numpy as np
from scipy import stats   # installed with scikit-learn

from src import config as C


def load_predictions(run, split="test", root=C.RESULTS_DIR):
    with np.load(Path(root) / "predictions" / f"{run}_{split}.npz") as d:   # plain str arrays, no pickle
        return {"files": d["files"].astype(str), "labels": d["labels"].astype(np.int64),
                "probs": d["probs"].astype(np.float32), "classes": d["classes"].astype(str)}


def macro_f1(y_true, y_pred, n_classes):
    """F1 per species = 2 TP / (n_true + n_pred), averaged over the species in y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes)
    n_true = np.bincount(y_true, minlength=n_classes)
    n_pred = np.bincount(y_pred, minlength=n_classes)
    present = n_true > 0
    return float(np.mean(2 * tp[present] / (n_true[present] + n_pred[present])))


def _scores(y_true, y_pred, n_classes):
    return {"accuracy": float(np.mean(y_true == y_pred)), "macro_f1": macro_f1(y_true, y_pred, n_classes)}


def _resamples(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, n)


def bootstrap_ci(labels, probs, n_classes, n_boot=1000, seed=0, alpha=0.05):
    """{metric: (point, lo, hi)} for accuracy and macro-F1, resampling files."""
    labels, pred = np.asarray(labels), np.asarray(probs).argmax(1)
    point = _scores(labels, pred, n_classes)
    boot = {m: [] for m in point}
    for idx in _resamples(len(labels), n_boot, seed):
        for m, v in _scores(labels[idx], pred[idx], n_classes).items():
            boot[m].append(v)
    q = [alpha / 2, 1 - alpha / 2]
    return {m: (point[m], *np.quantile(boot[m], q).tolist()) for m in point}


def paired_bootstrap(labels, probs_a, probs_b, n_classes, n_boot=1000, seed=0, alpha=0.05):
    """Macro-F1 of model a minus model b, with a percentile CI and a two-sided p-value."""
    labels = np.asarray(labels)
    pred_a, pred_b = np.asarray(probs_a).argmax(1), np.asarray(probs_b).argmax(1)
    delta = macro_f1(labels, pred_a, n_classes) - macro_f1(labels, pred_b, n_classes)
    boot = np.array([macro_f1(labels[i], pred_a[i], n_classes) - macro_f1(labels[i], pred_b[i], n_classes)
                     for i in _resamples(len(labels), n_boot, seed)])
    # share of resamples on each side of zero; +1 so p is never exactly 0 with a finite n_boot
    below = (np.sum(boot <= 0) + 1) / (n_boot + 1)
    above = (np.sum(boot >= 0) + 1) / (n_boot + 1)
    ci = tuple(np.quantile(boot, [alpha / 2, 1 - alpha / 2]).tolist())
    return {"delta_macro_f1": delta, "ci": ci, "p_value": float(min(1.0, 2 * min(below, above)))}


def mcnemar(labels, pred_a, pred_b):
    """b = files only a gets right, c = files only b gets right."""
    right_a = np.asarray(pred_a) == np.asarray(labels)
    right_b = np.asarray(pred_b) == np.asarray(labels)
    b, c = int(np.sum(right_a & ~right_b)), int(np.sum(~right_a & right_b))
    if b + c == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}
    if b + c < 25:
        stat = float(min(b, c))
        p = min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(stat, df=1)
    return {"b": b, "c": c, "statistic": stat, "p_value": float(p)}


def aggregate_seeds(results):
    """List of {metric: value} (one per seed) -> {metric: {mean, sd, n}}; missing / None values are skipped."""
    out = {}
    for m in dict.fromkeys(k for r in results for k in r):
        v = np.array([r[m] for r in results if r.get(m) is not None], dtype=float)
        v = v[~np.isnan(v)]
        out[m] = {"mean": float(v.mean()) if len(v) else float("nan"),
                  "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}
    return out
