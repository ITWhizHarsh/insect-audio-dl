"""Sanity checks on the official InsectSet459 train/validation/test split.

Checks that (1) no file appears in more than one split, (2) no iNaturalist /
xeno-canto observation is shared between splits (several files from one
observation in different splits would leak the same individual insect and
recording session into the test set), and (3) every species is present in
all three splits, and (4) measures recordist overlap: the official split is
stratified by species, not by recordist, so the same person can record the
same species in train and test. Such test files share microphone, site and
often the individual insect with training data and may score optimistically.

Usage:
    python -m src.data.verify_splits [--manifest data/raw/pilot/manifest.csv]
"""
import argparse
import json

import pandas as pd

from src import config as C


def verify(df):
    report = {"n_files": int(len(df)), "n_species": int(df.species_name.nunique())}
    report["files_per_split"] = df.subset.value_counts().to_dict()

    dup = df.groupby("file_name").subset.nunique()
    report["files_in_multiple_splits"] = int((dup > 1).sum())

    obs = df.dropna(subset=["observation"]).groupby("observation").subset.nunique()
    report["observations_shared_across_splits"] = int((obs > 1).sum())

    shared = obs[obs > 1].index.tolist()
    report["shared_observation_files"] = int(df.observation.isin(shared).sum())
    report["shared_observations"] = shared

    per_species = df.groupby("species_name").subset.nunique()
    report["species_missing_a_split"] = int((per_species < 3).sum())

    # recordist overlap, (contributor, species) pairs
    d = df.dropna(subset=["contributor"])
    pairs = d.groupby(["contributor", "species_name"]).subset.nunique()
    report["contributor_species_pairs"] = int(len(pairs))
    report["pairs_in_multiple_splits"] = int((pairs > 1).sum())
    train_pairs = set(map(tuple, d[d.subset == "Train"][["contributor", "species_name"]].values))
    for split in ("Validation", "Test"):
        part = d[d.subset == split]
        seen = [(c, s) in train_pairs for c, s in part[["contributor", "species_name"]].values]
        report[f"{split.lower()}_files_with_recordist_species_in_train"] = float(sum(seen) / max(len(part), 1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="subset manifest; default = full metadata")
    ap.add_argument("--min-recordings", type=int, default=0, help="restrict to species with >= n recordings")
    ap.add_argument("--out", default=None, help="write the report as json")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest or C.META_CSV)
    if args.min_recordings:
        counts = df.species_name.value_counts()
        df = df[df.species_name.isin(counts[counts >= args.min_recordings].index)]
    report = verify(df)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
