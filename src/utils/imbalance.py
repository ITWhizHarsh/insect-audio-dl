"""Class-imbalance strategies for the long-tailed species distribution.

* inverse-frequency weights:  w_c proportional to 1 / n_c
* effective-number weights (Cui et al., CVPR 2019): w_c proportional to
  (1 - beta) / (1 - beta ** n_c), which flattens out for very large
  classes instead of growing without bound like 1 / n_c
* a WeightedRandomSampler that draws every class equally often

Weights are normalised to mean 1 so the loss scale stays comparable to
unweighted training.
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
