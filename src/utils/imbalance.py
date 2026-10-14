"""Class-imbalance strategies for the long-tailed species distribution.

InsectSet459 is long-tailed: a few species have hundreds of recordings and
many have fewer than 25, so plain cross-entropy is pulled towards the
frequent species. The strategies compared are:

* inverse-frequency weights:  w_c proportional to 1 / n_c
* effective-number weights (Cui et al., CVPR 2019): w_c proportional to
  (1 - beta) / (1 - beta ** n_c), which flattens out for very large
  classes instead of growing without bound like 1 / n_c
* class-balanced sampling: a WeightedRandomSampler that draws every class
  equally often, so the batches are rebalanced instead of the loss
* logit adjustment (Menon et al., ICLR 2021): tau * log(prior_c) is added
  to the logits inside the training loss only. A rare true class then has
  to beat the frequent ones by a larger margin during training; at test
  time the plain logits are used, so the decision boundary ends up further
  from the rare classes

Weights are normalised to mean 1 so the loss scale stays comparable to
unweighted training.

Our 40-species benchmark subset is balanced by construction (about 30
training recordings per species), which hides the long tail. To test the
strategies in a controlled way, long_tail_subset builds a long-tailed
training set from it with the CIFAR-LT protocol (Cui et al. 2019; Cao et
al., NeurIPS 2019): class counts fall off exponentially from n_max to
n_max / ratio. Validation and test are left untouched.
"""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_counts(labels, n_classes):
    return np.bincount(labels, minlength=n_classes).astype(np.float64)


def class_weights(labels, n_classes, scheme="effective", beta=0.999):
    n = np.maximum(class_counts(labels, n_classes), 1)
    if scheme == "none":
        w = np.ones(n_classes)
    elif scheme == "inverse":
        w = 1.0 / n
    elif scheme == "effective":
        w = (1 - beta) / (1 - beta ** n)
    else:
        raise ValueError(scheme)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def balanced_sampler(labels, n_classes):
    n = np.maximum(class_counts(labels, n_classes), 1)
    per_sample = 1.0 / n[labels]
    return WeightedRandomSampler(torch.as_tensor(per_sample, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def long_tail_subset(index_df, ratio, seed=42):
    """Row positions of index_df (chunk rows) to keep for a long-tailed training set.

    Whole recordings are kept or dropped, never single chunks. Classes are
    ranked by a seeded random permutation, and the class at rank r keeps
    round(n_max * ratio ** (-r / (C - 1))) recordings (at least 1), where
    n_max is the smallest per-class recording count so every class can
    supply it. ratio=1 returns every row (the unmodified training set).
    """
    if ratio < 1:
        raise ValueError(ratio)
    if ratio == 1:
        return np.arange(len(index_df))
    files = index_df.groupby("file_name").label.first()     # one label per recording, sorted by name
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(files.values))
    n_max = files.value_counts().min()
    keep = []
    for rank, c in enumerate(order):
        n = max(1, int(round(n_max * ratio ** (-rank / max(len(order) - 1, 1)))))
        keep.extend(rng.choice(files.index[files.values == c], n, replace=False))
    return np.flatnonzero(index_df.file_name.isin(keep).values)


def logit_adjustment(counts, tau=1.0):
    """Per-class offsets tau * log(prior) for the logit-adjusted loss (Menon et al. 2021).

    counts can be class counts or prior probabilities; both are normalised to sum to 1.
    """
    prior = np.asarray(counts, dtype=np.float64)
    prior = prior / prior.sum()
    return torch.tensor(tau * np.log(np.maximum(prior, 1e-12)), dtype=torch.float32)


def adjusted_logits(logits, offsets):
    """Logits for the training loss only; evaluation uses the plain logits."""
    return logits + offsets.to(logits)
