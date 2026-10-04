"""Stream-and-delete feature extraction for InsectSet459.

Enables feature extraction without storing raw audio archives on disk:
1. Zenodo streaming mode: fetches byte ranges via HTTP into in-memory
   buffers (io.BytesIO), decodes, extracts 146-d features, and drops audio.
2. Local stream-and-delete mode: reads local files/archives, extracts features,
   and optionally deletes audio files immediately after extraction.
3. Resilient Checkpointing:
   - Per-split checkpoint: skips already completed splits if final arrays exist.
   - Per-recording checkpoint: saves intermediate chunk features to .ckpt_<split>/
     so that network disconnects resume instantly without re-downloading.

Usage:
    # 1. Local stream-and-delete test on 40-species dataset:
    python scripts/stream_features.py \
        --manifest data/meta/species40_manifest.csv \
        --audio-dir data/raw/species40 \
        --name species40_stream_test \
        --verify-against data/processed/species40

    # 2. Remote Zenodo in-memory streaming:
    python scripts/stream_features.py \
        --manifest data/meta/species40_manifest.csv \
        --source zenodo \
        --name species40_zenodo_stream \
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
from src import config as C  # noqa: E402
from src.data.preprocess import load_audio, chunk_audio  # noqa: E402
from src.features import handcrafted  # noqa: E402

ZENODO_URL_TEMPLATE = "https://zenodo.org/records/18554693/files/{}.zip?download=1"

thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        thread_local.session = s
    return thread_local.session


def read_zip_directory(split: str) -> dict:
    url = ZENODO_URL_TEMPLATE.format(split)
    with RemoteZip(url) as z:
        return {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}


def extract_features_from_audio(source, is_train: bool):
    """Load waveform from file path or io.BytesIO, slice into chunks, extract 146-d features."""
    y = load_audio(source, sr=C.SAMPLE_RATE)
    hop = C.TRAIN_HOP_SECONDS if is_train else C.EVAL_HOP_SECONDS
    cap = C.MAX_CHUNKS_TRAIN if is_train else C.MAX_CHUNKS_EVAL
    chunks = chunk_audio(y, hop, cap)
    hand = np.stack([handcrafted.extract(c) for c in chunks]).astype(np.float32)
    dur = len(y) / C.SAMPLE_RATE
    del y, chunks
    return hand, dur


def process_zenodo_member(url: str, info, is_train: bool, retries: int = 5):
    """Fetch range from Zenodo into RAM, decode, and extract features."""
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
        hand, dur = extract_features_from_audio(bio, is_train)
        del bio
        return hand, dur, None
    except Exception as exc:
        return None, None, f"ExtractError: {type(exc).__name__}: {exc}"


def process_local_file(path: Path, is_train: bool, delete_after: bool = False):
    """Load local file, extract features, and optionally delete raw audio immediately."""
    if not path.exists():
        return None, None, f"FileNotFound: {path}"
    try:
        hand, dur = extract_features_from_audio(path, is_train)
        if delete_after:
            try:
                path.unlink()
            except Exception as e:
                print(f"Warning: could not delete {path}: {e}")
        return hand, dur, None
    except Exception as exc:
        return None, None, f"ExtractError: {type(exc).__name__}: {exc}"


def run_pipeline(manifest_path: Path, output_dir: Path, source_type: str = "local",
                 audio_dir: Path = None, delete_after: bool = False,
                 workers: int = 4, splits=C.SPLITS, verify_against: Path = None):
    manifest = pd.read_csv(manifest_path)
    species = sorted(manifest.species_name.unique())
    label_of = {s: i for i, s in enumerate(species)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(json.dumps(species, indent=1))

    all_durations = {}
    all_failures = []
    parity_summary = {}

    for split in splits:
        part = manifest[manifest.subset == split].reset_index(drop=True)
        if len(part) == 0:
            continue
        is_train = (split == "Train")
        print(f"\n--- Split: {split} ({len(part)} files) [{source_type} mode] ---")

        # 1. Per-split checkpoint check
        final_hand_file = output_dir / f"{split}_hand.npy"
        final_index_file = output_dir / f"{split}_index.csv"
        if final_hand_file.exists() and final_index_file.exists():
            idx_df = pd.read_csv(final_index_file)
            if len(idx_df["file_name"].unique()) == len(part):
                print(f"  Split {split} already fully processed ({len(idx_df)} chunks). Skipping download.")
                if verify_against is not None:
                    exp_file = verify_against / f"{split}_hand.npy"
                    if exp_file.exists():
                        expected = np.load(exp_file)
                        actual = np.load(final_hand_file)
                        match = np.array_equal(actual, expected)
                        parity_summary[split] = {
                            "match": match,
                            "actual_shape": actual.shape,
                            "expected_shape": expected.shape,
                            "max_abs_diff": float(np.max(np.abs(actual - expected))),
                        }
                continue

        # 2. Per-recording checkpoint directory
        ckpt_dir = output_dir / f".ckpt_{split}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        durations = {}
        failures = []

        # Load existing checkpoints if resuming
        resumed_count = 0
        for idx, r in part.iterrows():
            ckpt_path = ckpt_dir / f"{idx}.npy"
            if ckpt_path.exists():
                try:
                    hand = np.load(ckpt_path)
                    results[idx] = (r["file_name"], r["species_name"], hand)
                    resumed_count += 1
                except Exception:
                    ckpt_path.unlink()  # corrupted checkpoint, recompute

        if resumed_count > 0:
            print(f"  Resumed {resumed_count}/{len(part)} files from checkpoint ({ckpt_dir.name}).")

        # Prepare remaining jobs
        if source_type == "zenodo":
            directory = read_zip_directory(split)
            url = ZENODO_URL_TEMPLATE.format(split)
            remaining_jobs = [
                (idx, r["file_name"], r["species_name"],
                 (url, directory[r["file_name"]], is_train))
                for idx, r in part.iterrows() if idx not in results
            ]
            worker_fn = lambda j: (j[0], j[1], j[2], process_zenodo_member(*j[3]))
        else:
            base = audio_dir if audio_dir else Path("data/raw")
            remaining_jobs = [
                (idx, r["file_name"], r["species_name"],
                 (base / split / r["file_name"], is_train, delete_after))
                for idx, r in part.iterrows() if idx not in results
            ]
            worker_fn = lambda j: (j[0], j[1], j[2], process_local_file(*j[3]))

        total = len(remaining_jobs)
        done = 0
        t0 = time.time()

        if total > 0:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(worker_fn, j) for j in remaining_jobs]
                for fut in as_completed(futures):
                    idx, fname, sp, (hand, dur, err) = fut.result()
                    done += 1
                    if err is not None:
                        failures.append({"file": fname, "error": err})
                        print(f"  [Error] {fname}: {err}", flush=True)
                    else:
                        results[idx] = (fname, sp, hand)
                        durations[fname] = dur
                        # Checkpoint individual recording to disk
                        np.save(ckpt_dir / f"{idx}.npy", hand)

                    if done % 100 == 0 or done == total:
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        print(f"  [{split} {resumed_count + done}/{len(part)}] {(resumed_count + done)/len(part)*100:.1f}% ({rate:.1f} files/s)", flush=True)

        # Reconstruct manifest row order
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

        index_df.to_csv(final_index_file, index=False)
        np.save(final_hand_file, concatenated_hand)
        all_durations.update(durations)
        all_failures.extend(failures)

        # Once split is fully assembled, clean up checkpoint files
        if len(results) == len(part):
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        print(f"  Saved {split}: {len(index_df)} chunks ({concatenated_hand.shape})")

        # Parity check if requested
        if verify_against is not None:
            exp_file = verify_against / f"{split}_hand.npy"
            if exp_file.exists():
                expected = np.load(exp_file)
                match = np.array_equal(concatenated_hand, expected)
                max_abs = float(np.max(np.abs(concatenated_hand - expected)))
                parity_summary[split] = {
                    "match": match,
                    "actual_shape": concatenated_hand.shape,
                    "expected_shape": expected.shape,
                    "max_abs_diff": max_abs,
                }
                print(f"  Parity vs {verify_against.name} ({split}): np.array_equal = {match}, max_abs_diff = {max_abs:.2e}")

    pd.DataFrame(list(all_durations.items()), columns=["file_name", "duration"]).to_csv(
        output_dir / "durations.csv", index=False
    )
    (output_dir / "failures.json").write_text(json.dumps(all_failures, indent=1))

    return parity_summary


def main():
    ap = argparse.ArgumentParser(description="Stream-and-delete feature extractor with checkpointing")
    ap.add_argument("--manifest", required=True, help="Manifest CSV")
    ap.add_argument("--source", choices=["local", "zenodo"], default="local",
                    help="Audio source: local files or Zenodo HTTP range streaming")
    ap.add_argument("--audio-dir", default=None, help="Directory containing audio splits (for local mode)")
    ap.add_argument("--delete-audio", action="store_true",
                    help="Delete audio file immediately after feature extraction (local mode only)")
    ap.add_argument("--name", default="species40_stream", help="Output directory under data/processed/")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"], help="Splits to process")
    ap.add_argument("--verify-against", default=None, help="Path to reference processed directory")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = C.PROCESSED_DIR / args.name
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    verify_dir = Path(args.verify_against) if args.verify_against else None

    t0 = time.time()
    parity = run_pipeline(
        manifest_path=manifest_path,
        output_dir=out_dir,
        source_type=args.source,
        audio_dir=audio_dir,
        delete_after=args.delete_audio,
        workers=args.workers,
        splits=args.splits,
        verify_against=verify_dir,
    )
    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s. Outputs in: {out_dir}")
    if parity:
        print("\n=== Parity Check Results ===")
        for s, res in parity.items():
            print(f"  {s}: np.array_equal = {res['match']}, max_abs_diff = {res['max_abs_diff']:.2e} (streamed {res['actual_shape']} vs ref {res['expected_shape']})")


if __name__ == "__main__":
    main()
