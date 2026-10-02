"""Common experimental protocol shared by every model in the comparison.

Every model reads its inputs from the same cached chunks and is scored by the
same metric code (src/eval/metrics.py), so differences in the comparison
table come from the architecture and not from the data handling.
"""
from pathlib import Path

SEED = 42

# audio
SAMPLE_RATE = 44_100          # keeps content up to 22.05 kHz (many Orthoptera sing above 16 kHz)
CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
TRAIN_HOP_SECONDS = 2.5       # 50% overlap when cutting training chunks
EVAL_HOP_SECONDS = 5.0        # no overlap for validation / test
MAX_CHUNKS_TRAIN = 8          # cap so long recordings don't dominate a species
MAX_CHUNKS_EVAL = 12

# log-mel spectrogram
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = SAMPLE_RATE // 2

# paths
ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "data/meta/InsectSet459_Train_Val_Test_Annotation.csv"
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"
RESULTS_DIR = ROOT / "results"

SPLITS = ("Train", "Validation", "Test")
