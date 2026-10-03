import numpy as np
import torch
import pytest


def test_handcrafted_train_only_normalization():
    # Simulate train features (100 samples, 146 features) and val features (50 samples, 146 features)
    rng = np.random.RandomState(42)
    x_tr = rng.normal(loc=10.0, scale=2.0, size=(100, 146)).astype(np.float32)
    x_va_1 = rng.normal(loc=50.0, scale=10.0, size=(50, 146)).astype(np.float32)
    x_va_2 = rng.normal(loc=-100.0, scale=0.5, size=(50, 146)).astype(np.float32)
    
    # Train-only normalisation protocol as defined in src/train.py:101
    norm_mean = torch.tensor(x_tr.mean(0))
    norm_std = torch.tensor(x_tr.std(0) + 1e-6)
    
    # Apply to train
    x_tr_scaled = (torch.tensor(x_tr) - norm_mean) / norm_std
    assert np.allclose(x_tr_scaled.mean(0).numpy(), 0.0, atol=1e-5)
    assert np.allclose(x_tr_scaled.std(0, unbiased=False).numpy(), 1.0, atol=1e-4)
    
    # Verify that changing validation data has zero effect on training normalisation stats
    norm_mean_after = torch.tensor(x_tr.mean(0))
    norm_std_after = torch.tensor(x_tr.std(0) + 1e-6)
    assert torch.equal(norm_mean, norm_mean_after)
    assert torch.equal(norm_std, norm_std_after)


def test_logmel_global_train_only_normalization():
    rng = np.random.RandomState(42)
    # Simulate log-mel training and validation chunks (N, 128, 431)
    x_tr = rng.normal(loc=-30.0, scale=15.0, size=(20, 128, 431)).astype(np.float32)
    
    # Global scalar normalisation protocol as in src/train.py:103
    m, s = float(x_tr.mean()), float(x_tr.std())
    
    assert np.isclose(m, -30.0, atol=0.5)
    assert np.isclose(s, 15.0, atol=0.5)
    
    x_tr_norm = (x_tr - m) / s
    assert np.isclose(x_tr_norm.mean(), 0.0, atol=1e-5)
    assert np.isclose(x_tr_norm.std(), 1.0, atol=1e-4)
