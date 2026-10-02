"""Log-mel spectrogram front end shared by the CNN, CRNN and (later) AST pipelines.

A 5 s chunk at 44.1 kHz with n_fft=1024 / hop=512 gives a 128 x 431
(mel bins x frames) image. Values are stored in dB (float16 on disk) and
standardised with train-set statistics at load time.
"""
import numpy as np
import librosa

from src import config as C

N_FRAMES = 1 + C.CHUNK_SAMPLES // C.HOP_LENGTH


def logmel(chunk, sr=C.SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)


def train_stats(specs):
    """Single global mean / std over the training set (per-bin stats overfit the pilot)."""
    return float(specs.mean()), float(specs.std())
