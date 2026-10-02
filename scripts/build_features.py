"""Run preprocessing once and cache every model's inputs.

For each recording in the manifest: load -> resample -> chunk, then compute
(a) the 146-d handcrafted vector (MLP) and (b) the 128 x 431 log-mel image
(CNN / CRNN). Outputs per split, in data/processed/<name>/:

    <split>_index.csv   one row per chunk: file_name, species_name, label, chunk
    <split>_hand.npy    float32 (N, 146)
    <split>_logmel.npy  float16 (N, 128, 431)

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
    path, is_train = args
    try:
        y = load_audio(path)
    except Exception as exc:
        return path, None, None, f"{type(exc).__name__}: {exc}"
    hop = C.TRAIN_HOP_SECONDS if is_train else C.EVAL_HOP_SECONDS
    cap = C.MAX_CHUNKS_TRAIN if is_train else C.MAX_CHUNKS_EVAL
    chunks = chunk_audio(y, hop, cap)
    hand = np.stack([handcrafted.extract(c) for c in chunks])
    mel = np.stack([logmel.logmel(c) for c in chunks]).astype(np.float16)
    return path, hand, mel, len(y) / C.SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", default="pilot")
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
        jobs = [(str(root / p), split == "Train") for p in part.path]
        rows, hands, mels = [], [], []
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
                mels.append(mel)
        pd.DataFrame(rows).to_csv(out / f"{split}_index.csv", index=False)
        np.save(out / f"{split}_hand.npy", np.concatenate(hands))
        np.save(out / f"{split}_logmel.npy", np.concatenate(mels))
        print(f"{split}: {part.shape[0]} files -> {len(rows)} chunks", flush=True)

    pd.Series(durations, name="seconds").to_csv(out / "durations.csv")
    (out / "failures.json").write_text(json.dumps(failures, indent=1))
    print(f"failed files: {len(failures)}")


if __name__ == "__main__":
    main()
