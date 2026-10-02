"""Audio preprocessing: decode -> mono -> resample -> normalise -> fixed-length chunks.

Recordings in InsectSet459 come from citizen-science sources with sample
rates between 8 kHz and 500 kHz, MP3 and WAV, mono and stereo, and durations
from under a second to several minutes. Every model needs fixed-size inputs
at one sample rate, so all files pass through the same steps here.
"""
import numpy as np
import librosa

from src import config as C


def load_audio(path, sr=C.SAMPLE_RATE):
    """Decode, downmix to mono, resample with soxr, remove DC, peak-normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True, res_type="soxr_hq")
    y = y - y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def chunk_audio(y, hop_seconds, max_chunks, chunk_samples=C.CHUNK_SAMPLES, sr=C.SAMPLE_RATE):
    """Cut a waveform into fixed-length chunks.

    Clips shorter than one chunk are tiled (repeated) rather than zero-padded:
    insect songs are repetitive, so tiling keeps the spectrogram statistics
    closer to a real recording than a block of silence would.
    Chunks are spread evenly across the recording when there are more
    candidate windows than `max_chunks`.
    """
    if len(y) < chunk_samples:
        reps = int(np.ceil(chunk_samples / max(len(y), 1)))
        return np.tile(y, reps)[:chunk_samples][None, :]

    hop = int(hop_seconds * sr)
    starts = np.arange(0, len(y) - chunk_samples + 1, hop)
    if len(starts) > max_chunks:
        starts = starts[np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)]
    return np.stack([y[s:s + chunk_samples] for s in starts])


def is_silent(chunk, threshold_db=-60.0):
    """True if a chunk carries essentially no signal (e.g. a gap between calls)."""
    rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
    return 20 * np.log10(rms) < threshold_db
