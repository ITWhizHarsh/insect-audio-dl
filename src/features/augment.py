"""Spectrogram-domain augmentation applied on the fly during training.

* random circular time shift - the call can start anywhere in a 5 s window
* SpecAugment (Park et al., 2019) - frequency and time masking
* mixup (Zhang et al., 2018) - convex mix of two examples and their labels

All operate on batched tensors of shape (B, 1, mels, frames) on the training
device, so they add almost no cost per step.
"""
import torch


def time_shift(x, max_frac=0.5):
    B, _, _, T = x.shape
    shifts = (torch.rand(B, device=x.device) * 2 - 1) * max_frac * T
    return torch.stack([torch.roll(x[i], int(s), dims=-1) for i, s in enumerate(shifts)])


def spec_augment(x, freq_masks=2, freq_width=16, time_masks=2, time_width=40):
    x = x.clone()
    B, _, F, T = x.shape
    fill = x.mean()
    for i in range(B):
        for _ in range(freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,)))
            f0 = int(torch.randint(0, max(F - w, 1), (1,)))
            x[i, :, f0:f0 + w, :] = fill
        for _ in range(time_masks):
            w = int(torch.randint(0, time_width + 1, (1,)))
            t0 = int(torch.randint(0, max(T - w, 1), (1,)))
            x[i, :, :, t0:t0 + w] = fill
    return x


def mixup(x, y_onehot, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)  # keep the first example dominant
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


def augment_batch(x, y_onehot, use_mixup=True):
    x = spec_augment(time_shift(x))
    if use_mixup:
        x, y_onehot = mixup(x, y_onehot)
    return x, y_onehot
