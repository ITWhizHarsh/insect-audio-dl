"""Run preprocessing once and cache every model's inputs.

For each recording in the manifest: load -> resample -> chunk, then compute
(a) the 146-d handcrafted vector (MLP) and (b) the 128 x 431 log-mel image
(CNN / CRNN). Outputs per split, in data/processed/<name>/:

    <split>_index.csv   one row per chunk: file_name, species_name, label, chunk
    <split>_hand.npy    float32 (N, 146)
    <split>_logmel.npy  float16 (N, 128, 431)

The log-mel arrays of the larger subsets do not fit in RAM on an 8 GB
machine (141 species: ~30k chunks x 128 x 431 x 2 bytes = 3.3 GB, and
np.concatenate would need a second copy). Each file's log-mels are therefore
appended to a raw temporary file on disk as soon as they are computed, and
copied into the final .npy with np.lib.format.open_memmap once the chunk
count is known. The result is the same file np.save would have written.

Manifests written on Windows store paths as data\raw\...; they are turned
into POSIX paths before being joined with the repository root.

Usage:
    python scripts/build_features.py --manifest data/raw/pilot/manifest.csv --name pilot
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.preprocess import load_audio, chunk_audio  # noqa: E402
from src.features import handcrafted, logmel  # noqa: E402


def process(args):
    path, is_train, hand_only = args
    try:
        y = load_audio(path)
    except Exception as exc:
        return path, None, None, f"{type(exc).__name__}: {exc}"
    hop = C.TRAIN_HOP_SECONDS if is_train else C.EVAL_HOP_SECONDS
    cap = C.MAX_CHUNKS_TRAIN if is_train else C.MAX_CHUNKS_EVAL
    chunks = chunk_audio(y, hop, cap)
    hand = np.stack([handcrafted.extract(c) for c in chunks])
    mel = None if hand_only else np.stack([logmel.logmel(c) for c in chunks]).astype(np.float16)
    return path, hand, mel, len(y) / C.SAMPLE_RATE


def posix(path):
    return path.replace("\\", "/")


def raw_to_npy(raw, path, shape, dtype, block=256):
    """Copy a raw file of stacked C-order arrays into a .npy file, block by block."""
    src = np.memmap(raw, dtype=dtype, mode="r", shape=shape)
    dst = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    for i in range(0, shape[0], block):
        dst[i:i + block] = src[i:i + block]
    dst.flush()
    del src, dst
    raw.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--hand-only", action="store_true", help="cache handcrafted features only (skip log-mel)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = C.ROOT
    manifest = pd.read_csv(root / args.manifest)
    species = sorted(manifest.species_name.unique())
    label_of = {s: i for i, s in enumerate(species)}
    out = C.PROCESSED_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(species, indent=1))

    durations, failures = {}, []
    for split in C.SPLITS:
        part = manifest[manifest.subset == split].reset_index(drop=True)
        jobs = [(str(root / posix(p)), split == "Train", args.hand_only) for p in part.path]
        rows, hands, mel_shape = [], [], None
        raw = out / f"{split}_logmel.part"
        mel_file = None if args.hand_only else open(raw, "wb")
        with ProcessPoolExecutor(args.workers) as pool:
            for (path, hand, mel, dur), (_, r) in zip(pool.map(process, jobs, chunksize=4), part.iterrows()):
                if hand is None:
                    failures.append({"file": r.file_name, "error": dur})
                    continue
                durations[r.file_name] = dur
                for k in range(len(hand)):
                    rows.append({"file_name": r.file_name, "species_name": r.species_name,
                                 "label": label_of[r.species_name], "chunk": k})
                hands.append(hand)
                if not args.hand_only:
                    mel_file.write(mel.tobytes())
                    mel_shape = mel.shape[1:]
        pd.DataFrame(rows).to_csv(out / f"{split}_index.csv", index=False)
        np.save(out / f"{split}_hand.npy", np.concatenate(hands))
        if not args.hand_only:
            mel_file.close()
            raw_to_npy(raw, out / f"{split}_logmel.npy", (len(rows), *mel_shape), np.float16)
        print(f"{split}: {part.shape[0]} files -> {len(rows)} chunks", flush=True)

    pd.Series(durations, name="seconds").to_csv(out / "durations.csv")
    (out / "failures.json").write_text(json.dumps(failures, indent=1))
    print(f"failed files: {len(failures)}")


if __name__ == "__main__":
    main()
