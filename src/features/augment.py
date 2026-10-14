"""Spectrogram-domain augmentation applied on the fly during training.

* random circular time shift - the call can start anywhere in a 5 s window
* SpecAugment (Park et al., 2019) - frequency and time masking
* mixup (Zhang et al., 2018) - convex mix of two examples and their labels

All operate on batched tensors of shape (B, 1, mels, frames) on the training
device. Each example still gets its own random shift and masks, but these are
built for the whole batch at once (an index grid for the shift, boolean masks
from broadcast comparisons for SpecAugment) instead of a Python loop over the
examples, so they add almost no cost per step on a GPU.

augment_batch has one switch per part so the training script can run the
augmentation ablation; with all three on it is the pilot behaviour.
"""
import torch


def time_shift(x, max_frac=0.5):
    B, _, _, T = x.shape
    shifts = ((torch.rand(B, device=x.device) * 2 - 1) * max_frac * T).long()   # truncated like int()
    # roll by s: output frame t takes input frame (t - s) mod T
    idx = (torch.arange(T, device=x.device) - shifts[:, None]) % T             # (B, T)
    return x.gather(-1, idx[:, None, None, :].expand_as(x))


def band_mask(B, n_masks, max_width, size, device):
    """(B, size) boolean mask holding n_masks bands per example, each of width uniform in [0, max_width]."""
    w = torch.randint(0, max_width + 1, (B, n_masks, 1), device=device)
    start = (torch.rand(B, n_masks, 1, device=device) * (size - w).clamp(min=1)).long()
    pos = torch.arange(size, device=device)
    return ((pos >= start) & (pos < start + w)).any(dim=1)


def spec_augment(x, freq_masks=2, freq_width=16, time_masks=2, time_width=40):
    B, _, F, T = x.shape
    f = band_mask(B, freq_masks, freq_width, F, x.device)
    t = band_mask(B, time_masks, time_width, T, x.device)
    mask = f[:, None, :, None] | t[:, None, None, :]           # (B, 1, F, T)
    return torch.where(mask, x.mean(), x)


def mixup(x, y_onehot, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)  # keep the first example dominant
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


def augment_batch(x, y_onehot, use_mixup=True, use_specaugment=True, use_shift=True):
    if use_shift:
        x = time_shift(x)
    if use_specaugment:
        x = spec_augment(x)
    if use_mixup:
        x, y_onehot = mixup(x, y_onehot)
    return x, y_onehot
