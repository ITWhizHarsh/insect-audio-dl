import numpy as np
import pytest
from src import config as C
from src.features import handcrafted, logmel


def test_handcrafted_dim_constant():
    assert handcrafted.FEATURE_DIM == 146


def test_extract_handcrafted_shape_and_validity():
    sr = C.SAMPLE_RATE
    t = np.linspace(0, 5.0, C.CHUNK_SAMPLES, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    
    feats = handcrafted.extract(y, sr=sr)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (146,)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_silent_handcrafted_features():
    sr = C.SAMPLE_RATE
    y = np.zeros(C.CHUNK_SAMPLES, dtype=np.float32)
    feats = handcrafted.extract(y, sr=sr)
    assert feats.shape == (146,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_logmel_shape_and_frames():
    sr = C.SAMPLE_RATE
    t = np.linspace(0, 5.0, C.CHUNK_SAMPLES, endpoint=False)
    y = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    mel = logmel.logmel(y, sr=sr)
    assert mel.shape == (128, 431)
    assert mel.dtype == np.float32
    assert not np.isnan(mel).any()
    assert not np.isinf(mel).any()
