"""Google Colab stream-and-delete feature extraction script.

Runs in Google Colab, mounts Google Drive, and streams audio directly from
Zenodo via HTTP range requests into memory, extracting 146-dimensional
handcrafted features directly onto Google Drive without downloading or
storing any raw audio files.

Resilient Checkpointing:
- Skips already completed splits on Drive.
- Checkpoints per-recording features into .ckpt_<split>/ on Drive so that if
  the Colab runtime disconnects, re-running resumes from the last completed file.

Peak Drive footprint: ~4.5 MB for 40 species (~46 MB for 141 species).
Zero audio files (.mp3/.wav) are saved to Drive or Colab disk.

Colab Usage:
    # 1. In Colab cell:
    from google.colab import drive
    drive.mount('/content/drive')

    # 2. Run feature extraction writing directly to Drive:
    !python scripts/colab_stream_features.py \
        --manifest data/meta/species40_manifest.csv \
        --drive-dir /content/drive/MyDrive/insect_dl/data/processed/species40 \
        --workers 4
"""
import argparse
import io
import json
import shutil
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src import config as C
    from src.data.preprocess import load_audio, chunk_audio
    from src.features import handcrafted
except ImportError:
    import librosa
    class C:
        SAMPLE_RATE = 44_100
        CHUNK_SAMPLES = int(44_100 * 5.0)
        TRAIN_HOP_SECONDS = 2.5
        EVAL_HOP_SECONDS = 5.0
        MAX_CHUNKS_TRAIN = 8
        MAX_CHUNKS_EVAL = 12
        FEATURE_DIM = 146

    def load_audio(source, sr=C.SAMPLE_RATE):
        y, _ = librosa.load(source, sr=sr, mono=True, res_type="soxr_hq")
        y = y - y.mean()
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak
        return y.astype(np.float32)

    def chunk_audio(y, hop_seconds, max_chunks, chunk_samples=C.CHUNK_SAMPLES, sr=C.SAMPLE_RATE):
        if len(y) < chunk_samples:
            reps = int(np.ceil(chunk_samples / max(len(y), 1)))
            return np.tile(y, reps)[:chunk_samples][None, :]
        hop = int(hop_seconds * sr)
        starts = np.arange(0, len(y) - chunk_samples + 1, hop)
        if len(starts) > max_chunks:
            starts = starts[np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)]
        return np.stack([y[s:s + chunk_samples] for s in starts])

ZENODO_URL_TEMPLATE = "https://zenodo.org/records/18554693/files/{}.zip?download=1"

thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def read_zip_directory(split: str) -> dict:
    url = ZENODO_URL_TEMPLATE.format(split)
    with RemoteZip(url) as z:
        return {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}


def fetch_and_extract(url: str, info, is_train: bool, retries: int = 5):
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size
    headers = {"Range": f"bytes={start}-{end}"}

    session = get_session()
    data = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep((attempt + 1) * 5)
                continue
            r.raise_for_status()
            buf = r.content
            n, m = struct.unpack("<HH", buf[26:30])
            payload = buf[30 + n + m: 30 + n + m + info.compress_size]
            data = zlib.decompress(payload, -15) if info.compress_type == 8 else payload
            assert len(data) == info.file_size
            break
        except Exception as exc:
            if attempt == retries - 1:
                return None, None, f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt * 3)

    if data is None:
        return None, None, "FetchFailed: 429 Too Many Requests"

    try:
        bio = io.BytesIO(data)
        del data
        y = load_audio(bio, sr=C.SAMPLE_RATE)
        del bio

        hop = C.TRAIN_HOP_SECONDS if is_train else C.EVAL_HOP_SECONDS
        cap = C.MAX_CHUNKS_TRAIN if is_train else C.MAX_CHUNKS_EVAL
        chunks = chunk_audio(y, hop, cap)
        hand = np.stack([handcrafted.extract(c) for c in chunks]).astype(np.float32)
        dur = len(y) / C.SAMPLE_RATE
        del y, chunks
        return hand, dur, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def process_split_to_drive(split: str, part: pd.DataFrame, label_of: dict,
                           directory: dict, drive_dir: Path, workers: int = 4):
    url = ZENODO_URL_TEMPLATE.format(split)
    is_train = (split == "Train")

    final_hand = drive_dir / f"{split}_hand.npy"
    final_idx = drive_dir / f"{split}_index.csv"
    if final_hand.exists() and final_idx.exists():
        idx_df = pd.read_csv(final_idx)
        if len(idx_df["file_name"].unique()) == len(part):
            print(f"[{split}] Already fully processed on Drive ({len(idx_df)} chunks). Skipping.")
            return idx_df, np.load(final_hand), {}, []

    ckpt_dir = drive_dir / f".ckpt_{split}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    durations = {}
    failures = []

    resumed = 0
    for idx, r in part.iterrows():
        cp = ckpt_dir / f"{idx}.npy"
        if cp.exists():
            try:
                results[idx] = (r["file_name"], r["species_name"], np.load(cp))
                resumed += 1
            except Exception:
                cp.unlink()

    if resumed > 0:
        print(f"[{split}] Resumed {resumed}/{len(part)} files from checkpoint on Drive.")

    remaining_jobs = [
        (idx, r["file_name"], r["species_name"], directory[r["file_name"]])
        for idx, r in part.iterrows() if idx not in results
    ]

    total = len(remaining_jobs)
    done = 0
    t0 = time.time()

    if total > 0:
        print(f"[{split}] Streaming {total} files directly to Drive ({workers} workers)...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(fetch_and_extract, url, info, is_train): (idx, fname, sp)
                for idx, fname, sp, info in remaining_jobs
            }
            for fut in as_completed(future_map):
                idx, fname, sp = future_map[fut]
                hand, dur, err = fut.result()
                done += 1
                if err is not None:
                    failures.append({"file": fname, "error": err})
                    print(f"  [Error] {fname}: {err}", flush=True)
                else:
                    results[idx] = (fname, sp, hand)
                    durations[fname] = dur
                    np.save(ckpt_dir / f"{idx}.npy", hand)

                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{split} {resumed + done}/{len(part)}] {(resumed + done)/len(part)*100:.1f}% ({rate:.1f} files/s)", flush=True)

    rows, hands = [], []
    for idx, r in part.iterrows():
        if idx not in results:
            continue
        fname, sp, hand = results[idx]
        for k in range(len(hand)):
            rows.append({
                "file_name": fname,
                "species_name": sp,
                "label": label_of[sp],
                "chunk": k,
            })
        hands.append(hand)

    concatenated_hand = np.concatenate(hands) if hands else np.empty((0, C.FEATURE_DIM), dtype=np.float32)
    index_df = pd.DataFrame(rows)

    drive_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(final_idx, index=False)
    np.save(final_hand, concatenated_hand)

    if len(results) == len(part):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"  Saved to Drive: {final_hand} ({concatenated_hand.shape})")
    return index_df, concatenated_hand, durations, failures


def main():
    ap = argparse.ArgumentParser(description="Colab stream-and-delete feature extractor with checkpointing")
    ap.add_argument("--manifest", default="data/meta/species40_manifest.csv")
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/insect_dl/data/processed/species40")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    drive_dir = Path(args.drive_dir)

    manifest = pd.read_csv(manifest_path)
    species = sorted(manifest.species_name.unique())
    label_of = {s: i for i, s in enumerate(species)}

    drive_dir.mkdir(parents=True, exist_ok=True)
    (drive_dir / "classes.json").write_text(json.dumps(species, indent=1))

    all_durations = {}
    all_failures = []
    t0 = time.time()

    for split in args.splits:
        part = manifest[manifest.subset == split].reset_index(drop=True)
        if len(part) == 0:
            continue
        directory = read_zip_directory(split)
        _, _, durations, failures = process_split_to_drive(
            split=split,
            part=part,
            label_of=label_of,
            directory=directory,
            drive_dir=drive_dir,
            workers=args.workers,
        )
        all_durations.update(durations)
        all_failures.extend(failures)

    pd.DataFrame(list(all_durations.items()), columns=["file_name", "duration"]).to_csv(
        drive_dir / "durations.csv", index=False
    )
    (drive_dir / "failures.json").write_text(json.dumps(all_failures, indent=1))

    elapsed = time.time() - t0
    print(f"\nAll splits saved to Google Drive in {elapsed:.1f}s at: {drive_dir}")


if __name__ == "__main__":
    main()
