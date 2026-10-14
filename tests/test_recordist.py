import numpy as np
import pandas as pd

from src.data.fetch_subset import choose_temperature_subset
from src.data.recordist import attach_metadata, seen_mask


def toy_meta():
    rows = [
        # file, species, contributor, subset, temperature
        ("t1.wav", "sp1", "Ann", "Train", 24.0),
        ("t2.wav", "sp2", "Bob", "Train", 60.0),       # data-entry error -> NaN
        ("t3.wav", "sp1", "Cat", "Train", -10.0),      # out of range -> NaN
        ("t4.wav", "sp2", None, "Train", np.nan),
        ("e1.wav", "sp1", "Ann", "Test", 50.0),        # same recordist and species as t1 -> seen
        ("e2.wav", "sp2", "Ann", "Test", -5.0),        # Ann never recorded sp2 in training -> unseen
        ("e3.wav", "sp2", None, "Test", np.nan),       # unknown recordist -> unseen
        ("e4.wav", "sp1", "Cat", "Test", 20.0),        # Cat's t3 is not in the training index -> unseen
        ("e5.wav", "sp2", "Bob", "Test", 30.0),        # Bob recorded sp2 in t2 -> seen
    ]
    meta = pd.DataFrame(rows, columns=["file_name", "species_name", "contributor", "subset", "temperature"])
    meta["group"] = "Orthoptera"
    return meta


def toy_index(files, chunks=(2, 1, 3, 1, 2, 2)):
    """Chunk index with several rows per file, as written by scripts/build_features.py."""
    rows = []
    for f, n in zip(files, chunks):
        species = toy_meta().set_index("file_name").species_name.get(f, "sp1")
        rows += [{"file_name": f, "species_name": species, "label": int(species[-1]) - 1, "chunk": k}
                 for k in range(n)]
    return pd.DataFrame(rows)


def test_attach_metadata_cleans_temperature_and_keeps_rows():
    index = toy_index(["t1.wav", "t2.wav", "t3.wav", "e1.wav", "e2.wav", "missing.wav"])
    before = index.copy()
    out = attach_metadata(index, toy_meta())
    pd.testing.assert_frame_equal(index, before)                     # input untouched
    assert list(out.index) == list(index.index) and len(out) == len(index)
    per_file = out.groupby("file_name", sort=False).first()
    assert per_file.temperature["t1.wav"] == 24.0
    assert np.isnan(per_file.temperature["t2.wav"])                  # 60 C
    assert np.isnan(per_file.temperature["t3.wav"])                  # -10 C
    assert per_file.temperature["e1.wav"] == 50.0                    # range ends are valid
    assert per_file.temperature["e2.wav"] == -5.0
    assert per_file.contributor["e2.wav"] == "Ann"
    assert per_file.group["t1.wav"] == "Orthoptera"
    missing = out[out.file_name == "missing.wav"]
    assert missing[["contributor", "group", "temperature"]].isna().all().all()


def test_seen_mask_pairs_recordist_with_species():
    train = toy_index(["t1.wav", "t2.wav", "t4.wav"])
    test = toy_index(["e3.wav", "e1.wav", "e2.wav", "e4.wav", "e5.wav", "missing.wav"])
    test = pd.concat([test, test.iloc[[0, 2]]], ignore_index=True)    # repeated rows do not add files
    seen = seen_mask(test, train, toy_meta())
    assert list(seen.index) == ["e3.wav", "e1.wav", "e2.wav", "e4.wav", "e5.wav", "missing.wav"]
    assert seen.dtype == bool
    assert seen.to_dict() == {"e3.wav": False, "e1.wav": True, "e2.wav": False,
                              "e4.wav": False, "e5.wav": True, "missing.wav": False}


def test_seen_mask_missing_contributor_never_matches():
    # a training file without a contributor must not make an unknown-recordist test file "seen"
    train = toy_index(["t4.wav"])
    test = toy_index(["e3.wav"])
    assert not seen_mask(test, train, toy_meta()).any()


def labelled_rows(species, group, n_train, n_val, n_test, temperature=25.0):
    rows = []
    for subset, n in (("Train", n_train), ("Validation", n_val), ("Test", n_test)):
        rows += [{"file_name": f"{species}_{subset}_{i}.wav", "species_name": species, "group": group,
                  "subset": subset, "temperature": temperature} for i in range(n)]
    return rows


def test_choose_temperature_subset():
    rows = (labelled_rows("ok", "Orthoptera", 20, 3, 3)
            + labelled_rows("ok_unlabelled", "Orthoptera", 5, 2, 2, temperature=np.nan)
            + labelled_rows("few_test", "Orthoptera", 25, 4, 2)
            + labelled_rows("no_val", "Orthoptera", 30, 0, 5)        # no Validation rows at all
            + labelled_rows("cicada", "Cicadidae", 30, 5, 5))
    meta = pd.DataFrame(rows)
    meta.loc[meta.species_name == "ok_unlabelled", "species_name"] = "ok"   # unlabelled files of "ok"
    outlier = labelled_rows("outlier", "Orthoptera", 20, 3, 3)
    outlier[0]["temperature"] = 60.0                                       # only 19 valid Train files
    meta = pd.concat([meta, pd.DataFrame(outlier)], ignore_index=True)

    chosen = choose_temperature_subset(meta)
    assert set(chosen.species_name) == {"ok"}
    assert len(chosen) == 26 and chosen.temperature.notna().all()
    assert chosen.subset.value_counts().to_dict() == {"Train": 20, "Validation": 3, "Test": 3}

    looser = choose_temperature_subset(meta, min_train=19, min_eval=2)
    assert set(looser.species_name) == {"ok", "few_test", "outlier"}
    assert looser.temperature.between(-5, 50).all()
