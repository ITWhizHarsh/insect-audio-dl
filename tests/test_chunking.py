import numpy as np
import pytest
from src import config as C
from src.data.preprocess import chunk_audio, is_silent


def test_short_audio_tiling():
    sr = C.SAMPLE_RATE
    # 2 seconds of synthetic audio (< 5s chunk)
    y_short = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 2 * sr, endpoint=False)).astype(np.float32)
    chunks = chunk_audio(y_short, hop_seconds=2.5, max_chunks=8)
    assert chunks.shape == (1, C.CHUNK_SAMPLES)
    # Check that it is tiled, not zero-padded
    assert np.allclose(chunks[0, :len(y_short)], y_short)
    assert np.allclose(chunks[0, len(y_short):2 * len(y_short)], y_short)
    assert not np.allclose(chunks[0, len(y_short):], 0.0)


def test_exact_chunk_length():
    sr = C.SAMPLE_RATE
    y_exact = np.ones(C.CHUNK_SAMPLES, dtype=np.float32) * 0.5
    chunks = chunk_audio(y_exact, hop_seconds=2.5, max_chunks=8)
    assert chunks.shape == (1, C.CHUNK_SAMPLES)
    assert np.allclose(chunks[0], 0.5)


def test_medium_audio_chunk_count():
    sr = C.SAMPLE_RATE
    # 15 seconds: candidates = (15 - 5) / 2.5 + 1 = 5 chunks
    y_15s = np.random.randn(15 * sr).astype(np.float32)
    chunks = chunk_audio(y_15s, hop_seconds=2.5, max_chunks=8)
    assert chunks.shape == (5, C.CHUNK_SAMPLES)


def test_chunk_cap_limits():
    sr = C.SAMPLE_RATE
    # 120 seconds of audio
    y_120s = np.random.randn(120 * sr).astype(np.float32)
    
    # Train cap = 8
    chunks_tr = chunk_audio(y_120s, hop_seconds=C.TRAIN_HOP_SECONDS, max_chunks=C.MAX_CHUNKS_TRAIN)
    assert chunks_tr.shape == (C.MAX_CHUNKS_TRAIN, C.CHUNK_SAMPLES)
    
    # Eval cap = 12
    chunks_ev = chunk_audio(y_120s, hop_seconds=C.EVAL_HOP_SECONDS, max_chunks=C.MAX_CHUNKS_EVAL)
    assert chunks_ev.shape == (C.MAX_CHUNKS_EVAL, C.CHUNK_SAMPLES)


def test_silence_detection():
    sr = C.SAMPLE_RATE
    silent = np.zeros(C.CHUNK_SAMPLES, dtype=np.float32)
    active = np.sin(2 * np.pi * 1000 * np.linspace(0, 5.0, C.CHUNK_SAMPLES, endpoint=False)).astype(np.float32)
    
    assert is_silent(silent, threshold_db=-60.0)
    assert not is_silent(active, threshold_db=-60.0)
