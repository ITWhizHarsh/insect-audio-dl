"""Who records temperature? Confounding audit before the conditioning experiment.

If temperature is only entered by a few recordists, a model given temperature
could learn "this recordist / region" instead of the physiological effect of
temperature on song. This script measures how concentrated the temperature
field is and how much of it is shared between train and test recordists.

Usage:
    python analysis/temperature_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def main():
    df = pd.read_csv(C.META_CSV)
    raw = df.temperature.notna().sum()
    out_of_range = df[(df.temperature < -5) | (df.temperature > 50)].temperature.tolist()
    df.loc[(df.temperature < -5) | (df.temperature > 50), "temperature"] = np.nan
    orth = df[df.group == "Orthoptera"].dropna(subset=["contributor"])
    t = orth.dropna(subset=["temperature"])

    per_rec = orth.groupby("contributor").temperature.apply(lambda s: s.notna().mean())
    n_rec = orth.groupby("contributor").size()
    active = per_rec[n_rec >= 10]          # recordists with at least 10 Orthoptera files
    by_rec = t.contributor.value_counts()

    report = {
        "temperature_values_raw": int(raw),
        "out_of_range_removed": out_of_range,
        "files_with_valid_temperature": int(len(t)),
        "orthoptera_files": int(len(orth)),
        "orthoptera_coverage": float(len(t) / len(orth)),
        "recordists_with_any_temperature": int(by_rec.size),
        "top1_recordist_share": float(by_rec.iloc[0] / len(t)),
        "top5_recordists_share": float(by_rec.head(5).sum() / len(t)),
        "recordists_>=10_files": int(active.size),
        "of_which_all_or_nothing": int(((active > 0.9) | (active < 0.1)).sum()),
        "temperature_by_split": t.subset.value_counts().to_dict(),
        "train_mean_c": float(t[t.subset == "Train"].temperature.mean()),
        "train_std_c": float(t[t.subset == "Train"].temperature.std()),
    }
    # species with temperature from a single recordist cannot separate the two effects
    rec_per_species = t.groupby("species_name").contributor.nunique()
    report["species_with_temperature"] = int(rec_per_species.size)
    report["species_temperature_from_one_recordist"] = int((rec_per_species == 1).sum())

    path = C.RESULTS_DIR / "metrics" / "temperature_audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
