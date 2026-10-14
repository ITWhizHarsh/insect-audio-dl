"""Recordist metadata for chunk indexes, and the seen / unseen recordist test split.

The official InsectSet459 split is stratified by species, not by recordist,
and src/data/verify_splits.py found that most test files come from a person
who also recorded the same species in the training split. Such a file shares
microphone, site and often the individual insect with training data, so a
model can score well on it without having learned the species' song in
general. Scoring the test set separately on "recordist seen" and "recordist
unseen" files shows how much of a model's accuracy depends on that overlap.

Temperatures outside [-5, 50] C are treated as missing, as in
analysis/temperature_audit.py: the two values of 60 in the metadata are data
entry errors, not field conditions.
"""
import numpy as np
import pandas as pd

from src import config as C

TEMP_RANGE = (-5, 50)


def clean_temperature(meta):
    """Copy of the metadata with implausible temperatures set to NaN."""
    meta = meta.copy()
    bad = (meta.temperature < TEMP_RANGE[0]) | (meta.temperature > TEMP_RANGE[1])
    meta.loc[bad, "temperature"] = np.nan
    return meta


def load_meta(meta=None):
    meta = pd.read_csv(C.META_CSV) if meta is None else meta
    return clean_temperature(meta).set_index("file_name")


def attach_metadata(index_df, meta=None):
    """Copy of a chunk index with contributor, group and temperature added per file.

    Rows keep their order and index; files missing from the metadata get NaN.
    """
    by_file = load_meta(meta)
    out = index_df.copy()
    for col in ("contributor", "group", "temperature"):
        out[col] = out.file_name.map(by_file[col])
    return out


def seen_mask(eval_index, train_index, meta=None):
    """Per evaluation file: was its (recordist, species) pair in the training files?

    Returns a boolean Series indexed by the unique file names of eval_index,
    in order of first appearance (the same order as metrics.file_level).
    A file with no contributor in the metadata cannot be matched to anyone,
    so it counts as unseen.
    """
    meta = pd.read_csv(C.META_CSV) if meta is None else meta
    train = attach_metadata(train_index.drop_duplicates("file_name"), meta).dropna(subset=["contributor"])
    train_pairs = set(zip(train.contributor, train.species_name))
    files = attach_metadata(eval_index.drop_duplicates("file_name"), meta)
    seen = [pd.notna(c) and (c, s) in train_pairs for c, s in zip(files.contributor, files.species_name)]
    return pd.Series(seen, index=files.file_name.values, name="recordist_seen")
