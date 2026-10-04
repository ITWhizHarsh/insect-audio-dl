"""Fetch a species subset of InsectSet459 without downloading the full archives.

InsectSet459 is distributed as three zip files on Zenodo (Train 51 GB,
Validation 16 GB, Test 16 GB). Zenodo serves them with HTTP range support, so
we read each zip's central directory once and then request only the byte
range of every member we need. One range request per recording, raw-deflate
decoded locally.

Selection rule (deterministic, seed fixed):
  * species with >= MIN_RECORDINGS recordings in the official annotation CSV
  * the N most frequent such species, balanced between Orthoptera and Cicadidae
  * per species and per official split, up to a cap of recordings whose
    compressed size is <= MAX_MB (keeps the pilot download small)

Usage:
    python -m src.data.fetch_subset --n-species 12 --out data/raw/pilot
"""
import argparse
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from remotezip import RemoteZip

ZENODO = "https://zenodo.org/records/18554693/files/{}.zip?download=1"
SPLITS = ["Train", "Validation", "Test"]
CAPS = {"Train": 30, "Validation": 10, "Test": 10}
META = Path("data/meta/InsectSet459_Train_Val_Test_Annotation.csv")


def choose_species(meta, n_species=12, min_recordings=40, threshold_mode=False):
    counts = meta.groupby(["species_name", "group"]).size().reset_index(name="n")
    counts = counts[counts.n >= min_recordings].sort_values("n", ascending=False)
    if threshold_mode:
        return sorted(counts.species_name.unique().tolist())
    per_group = n_species // 2
    chosen = []
    for group in ["Orthoptera", "Cicadidae"]:
        chosen += counts[counts.group == group].species_name.head(per_group).tolist()
    return sorted(chosen)


def read_directory(split):
    """Return {member basename: ZipInfo} using only the zip central directory."""
    with RemoteZip(ZENODO.format(split)) as z:
        return {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}


def fetch_member(url, info, dest, retries=4):
    # local header is 30 bytes + name + extra; extra may differ from the
    # central directory copy, so over-read a little and parse it
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            buf = r.content
            n, m = struct.unpack("<HH", buf[26:30])
            payload = buf[30 + n + m: 30 + n + m + info.compress_size]
            data = zlib.decompress(payload, -15) if info.compress_type == 8 else payload
            assert len(data) == info.file_size, "size mismatch"
            dest.write_bytes(data)
            return dest.name, len(data)
        except Exception as exc:  # network hiccups / 429s from Zenodo
            if attempt == retries - 1:
                raise RuntimeError(f"{dest.name}: {exc}") from exc
            time.sleep(2 ** attempt * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-species", type=int, default=12)
    ap.add_argument("--min-recordings", type=int, default=40)
    ap.add_argument("--threshold-mode", action="store_true",
                    help="Select all species with >= min-recordings with no per-species cap or group truncation")
    ap.add_argument("--max-mb", type=float, default=3.0)
    ap.add_argument("--no-max-mb", action="store_true",
                    help="Do not filter recordings by compressed size")
    ap.add_argument("--out", default="data/raw/pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caps", type=int, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"),
                    help="max recordings per species in each split (default 30 10 10)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print exact compressed size per split from central directories without downloading audio")
    args = ap.parse_args()
    caps = dict(zip(SPLITS, args.caps)) if args.caps else CAPS

    meta = pd.read_csv(META)
    species = choose_species(meta, args.n_species, args.min_recordings, threshold_mode=args.threshold_mode)
    print(f"{len(species)} species:", ", ".join(species))

    rows, jobs = [], []
    for split in SPLITS:
        directory = read_directory(split)
        part = meta[(meta.subset == split) & meta.species_name.isin(species)].copy()
        part = part[part.file_name.isin(directory)]
        part["compress_mb"] = part.file_name.map(lambda f: directory[f].compress_size / 1e6)
        if not (args.threshold_mode or args.no_max_mb):
            part = part[part.compress_mb <= args.max_mb]
        if not args.threshold_mode:
            part = (part.sample(frac=1, random_state=args.seed)
                        .groupby("species_name").head(caps[split]))
        elif args.caps:
            part = (part.sample(frac=1, random_state=args.seed)
                        .groupby("species_name").head(caps[split]))

        if not args.dry_run:
            out_dir = Path(args.out) / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in part.file_name:
                dest = out_dir / f
                if not dest.exists():
                    jobs.append((ZENODO.format(split), directory[f], dest))
        rows.append(part)
        print(f"{split}: {len(part)} files, {part.compress_mb.sum():.1f} MB compressed ({part.compress_mb.sum() / 1024:.2f} GB)")

    total_files = sum(len(p) for p in rows)
    total_mb = sum(p.compress_mb.sum() for p in rows)
    print(f"Total: {total_files} files, {total_mb:.1f} MB ({total_mb / 1024:.2f} GB) compressed")

    if args.dry_run:
        print("[dry-run] Audio download skipped.")
        return

    manifest = pd.concat(rows)
    manifest["path"] = [str(Path(args.out) / s / f) for s, f in zip(manifest.subset, manifest.file_name)]
    manifest.to_csv(Path(args.out) / "manifest.csv", index=False)

    done, t0, total = 0, time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_member, *j) for j in jobs]
        for fut in as_completed(futures):
            name, size = fut.result()
            done += 1
            total += size
            if done % 25 == 0 or done == len(jobs):
                rate = total / 1e6 / (time.time() - t0)
                print(f"[{done}/{len(jobs)}] {total / 1e6:.0f} MB, {rate:.2f} MB/s", flush=True)
    print("manifest:", Path(args.out) / "manifest.csv")


if __name__ == "__main__":
    main()
