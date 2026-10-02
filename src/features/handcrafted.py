"""Handcrafted acoustic descriptors for the MLP baseline.

Each 5 s chunk becomes one fixed-length vector of summary statistics
(mean and standard deviation over time) of frame-level descriptors:

  * 20 MFCCs + delta + delta-delta           -> 120 values
  * spectral centroid, bandwidth, roll-off,
    flatness, zero-crossing rate, RMS energy  -> 12 values
  * 7-band spectral contrast                  -> 14 values

146 dimensions in total. The MLP never sees the time-frequency image, only
these statistics, which is what makes it the "no learned features" reference
point of the comparison.
"""
import numpy as np
import librosa

from src import config as C


def _stats(m):
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def extract(chunk, sr=C.SAMPLE_RATE):
    S = np.abs(librosa.stft(chunk, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    spectral = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(chunk, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH),
        librosa.feature.rms(S=S, frame_length=C.N_FFT),
    ])
    # centroid / bandwidth / roll-off are in Hz; scale to kHz so magnitudes are comparable
    spectral[:3] /= 1000.0
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=200.0)

    return np.concatenate([_stats(mfcc), _stats(d1), _stats(d2),
                           _stats(spectral), _stats(contrast)]).astype(np.float32)


FEATURE_DIM = 146
