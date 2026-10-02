"""Sanity checks on the official InsectSet459 train/validation/test split.

Checks that (1) no file appears in more than one split, (2) no iNaturalist /
xeno-canto observation is shared between splits (several files from one
observation in different splits would leak the same individual insect and
recording session into the test set), and (3) every species is present in
all three splits.

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

    per_species = df.groupby("species_name").subset.nunique()
    report["species_missing_a_split"] = int((per_species < 3).sum())
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="subset manifest; default = full metadata")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest or C.META_CSV)
    print(json.dumps(verify(df), indent=2))


if __name__ == "__main__":
    main()
