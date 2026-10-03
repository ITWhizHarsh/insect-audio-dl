import pandas as pd
import pytest
from src import config as C
from src.data.verify_splits import verify


def test_meta_csv_exists():
    assert C.META_CSV.exists(), f"Metadata CSV missing at {C.META_CSV}"


def test_split_disjointness_and_species_coverage_141():
    df = pd.read_csv(C.META_CSV)
    counts = df.groupby("species_name").size()
    keep = counts[counts >= 40].index
    df_141 = df[df.species_name.isin(keep)].copy()
    
    report = verify(df_141)
    
    # Exactly 141 species and 20,319 files
    assert report["n_species"] == 141
    assert report["n_files"] == 20319
    
    # 0 files in multiple splits
    assert report["files_in_multiple_splits"] == 0
    
    # Every species present in all 3 splits
    assert report["species_missing_a_split"] == 0
    
    # Exact split counts
    assert report["files_per_split"]["Train"] == 12166
    assert report["files_per_split"]["Validation"] == 4083
    assert report["files_per_split"]["Test"] == 4070
    
    # Exactly 1 shared observation (the Cicada Central DOI across 6 files)
    assert report["observations_shared_across_splits"] == 1
    assert report["shared_observation_files"] == 6
