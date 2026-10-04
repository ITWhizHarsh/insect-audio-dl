# 40-Species Stage: MLP Baseline Benchmark

40-species stage, handcrafted MLP, CPU. Not comparable to the 12-species pilot or the published 459-class baselines. Test numbers were not used for any tuning.

## Dataset Selection & Scope
- **Taxonomic Selection Rule:** Top 20 Orthoptera + top 20 Cicadidae species by recording count in metadata (among species with >= 40 recordings). All 40 species are a strict subset of the target 141-species dataset.
- **Per-Species Caps:** Capped at 30 Train, 10 Validation, and 10 Test recordings per species (with <= 3 MB audio download cap).
- **Per-Split File & Chunk Counts:**
  - **Train:** 1,193 files (5,196 chunks)
  - **Validation:** 400 files (1,301 chunks)
  - **Test:** 398 files (1,219 chunks)
  - **Total:** 1,991 files (7,716 chunks)
- **Manifest Reference:** `data/meta/species40_manifest.csv`

## Benchmark Results (Seeds 42–46)

| Seed | Best Epoch | Train Time (s) | Val Macro-F1 (File) | Val Acc (File) | Test Macro-F1 (File) | Test Acc (File) | Result JSON |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **42** | 51 | 50.6s | 0.6440 | 0.6475 | 0.5975 | 0.6030 | `results/metrics/mlp_species40_s42.json` |
| **43** | 18 | 53.8s | 0.6316 | 0.6375 | 0.5971 | 0.6030 | `results/metrics/mlp_species40_s43.json` |
| **44** | 82 | 71.0s | 0.6516 | 0.6525 | 0.5799 | 0.5905 | `results/metrics/mlp_species40_s44.json` |
| **45** | 112 | 49.7s | 0.6462 | 0.6500 | 0.6196 | 0.6307 | `results/metrics/mlp_species40_s45.json` |
| **46** | 77 | 49.3s | 0.6445 | 0.6500 | 0.5985 | 0.6080 | `results/metrics/mlp_species40_s46.json` |

### Summary Statistics (Mean ± SD across 5 seeds)
- **Validation File Macro-F1:** 0.6436 ± 0.0074 (min 0.6316, max 0.6516)
- **Test File Macro-F1:** 0.5985 ± 0.0141 (min 0.5799, max 0.6196)
- **Test File Accuracy:** 0.6070 ± 0.0147 (min 0.5905, max 0.6307)

## Code Dependency & Reproduction Commands
*Note: Reproduction requires the `--hand-only` feature caching flag and dry-run/threshold subset fetching implemented in branch `feat/ap-subset-threshold` (PR #2).*

```bash
# 1. Dry run verification (confirms 1,991 files, 1.03 GB)
python -m src.data.fetch_subset --n-species 40 --dry-run

# 2. Download audio subset
python -m src.data.fetch_subset --n-species 40 --out data/raw/species40 --workers 8

# 3. Cache 146-d handcrafted features (skips log-mel images to save disk)
python scripts/build_features.py --manifest data/raw/species40/manifest.csv --name species40 --hand-only

# 4. Train MLP baseline across seeds 42-46 (CPU execution)
python -m src.train --model mlp --data species40 --epochs 150 --batch 64 --wd 1e-4 --seed <SEED> --tag _s<SEED>
```

## Environment Block
```json
{
  "os": "Windows-11-10.0.26200-SP0",
  "python": "3.13.5",
  "torch": "2.12.0+cpu",
  "numpy": "2.3.3",
  "pandas": "2.3.2",
  "sklearn": "1.8.0",
  "librosa": "1.0.0",
  "soundfile": "0.14.0",
  "libsndfile": "1.2.2",
  "soxr": "1.1.0",
  "cuda_available": false
}
```
