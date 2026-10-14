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

## Re-run under the final protocol (Apple M2, 14 Oct 2026)

The five seeds were re-run with the final shared training loop (`src/train.py`) so that the
MLP has the same outputs as every other model: saved file-level predictions and separate
scores for recordist-seen / recordist-unseen test files. Features were re-extracted on macOS
from the same manifest (identical recordings, chunk counts and order: 5,196 / 1,301 / 1,219).
Differences from the Windows run come from library versions (librosa 0.11 vs 1.0, numpy,
torch 2.5 vs 2.12) and the new batch-order generator, not from the protocol.

| Seed | Best Epoch | Train Time | Val Macro-F1 | Test Macro-F1 | Test Acc | Test F1 recordist-seen (221 files) | Test F1 recordist-unseen (177 files) | Windows Test Macro-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 50 | 151.6s | 0.6403 | 0.6164 | 0.6256 | 0.6101 | 0.5765 | 0.5975 |
| **43** | 109 | 112.2s | 0.6424 | 0.6109 | 0.6206 | 0.6282 | 0.5318 | 0.5971 |
| **44** | 134 | 137.4s | 0.6380 | 0.6004 | 0.6080 | 0.6121 | 0.4913 | 0.5799 |
| **45** | 52 | 115.3s | 0.6182 | 0.6020 | 0.6106 | 0.6153 | 0.5202 | 0.6196 |
| **46** | 113 | 151.3s | 0.6372 | 0.6151 | 0.6231 | 0.6282 | 0.5408 | 0.5985 |

- **Validation file macro-F1:** 0.6352 ± 0.0097 (Windows: 0.6436 ± 0.0074)
- **Test file macro-F1:** 0.6090 ± 0.0074 (Windows: 0.5985 ± 0.0141)
- **Test file accuracy:** 0.6176 ± 0.0078 (Windows: 0.6070 ± 0.0147)
- **Test macro-F1, recordist-seen / unseen:** 0.6188 / 0.5321. Test files whose
  recordist also recorded that species for training are easier, as the split audit predicted.

The Mac and Windows means differ by +0.0105 test macro-F1, about
0.7 Windows seed standard deviations.

### Classical baselines on the same features (`analysis/classical_baselines.py`)

| Model | Val Macro-F1 | Test Macro-F1 | Test Acc |
|:---|:---:|:---:|:---:|
| Logistic regression | 0.5641 | 0.5541 | 0.5628 |
| RBF SVM | 0.5409 | 0.5548 | 0.5578 |
| Random forest | 0.5428 | 0.5249 | 0.5352 |
| MLP (5 seeds) | 0.6352 | 0.6090 | 0.6176 |

### Which handcrafted features matter? (`analysis/mlp_feature_ablation.py`, seeds 42-44)

Same MLP and training settings, trained on one feature group alone or on all groups but one.
MFCC statistics carry most of the species information; the six spectral-shape descriptors
and the spectral contrast add nothing measurable once the MFCCs are present (dropping either
leaves test macro-F1 within one seed standard deviation), and the delta features add a
little.

| Features | Dimensions | Val Macro-F1 | Test Macro-F1 | Test Acc |
|:---|:---:|:---:|:---:|:---:|
| all features | 146 | 0.6425 ± 0.0119 | 0.5989 ± 0.0133 | 0.6089 ± 0.0126 |
| only mfcc | 40 | 0.5789 ± 0.0080 | 0.5476 ± 0.0033 | 0.5553 ± 0.0050 |
| only delta | 40 | 0.4222 ± 0.0059 | 0.3868 ± 0.0125 | 0.3928 ± 0.0102 |
| only delta2 | 40 | 0.2958 ± 0.0095 | 0.2555 ± 0.0086 | 0.2638 ± 0.0091 |
| only spectral | 12 | 0.3152 ± 0.0099 | 0.2671 ± 0.0210 | 0.2797 ± 0.0182 |
| only contrast | 14 | 0.2605 ± 0.0033 | 0.2441 ± 0.0018 | 0.2496 ± 0.0038 |
| without mfcc | 106 | 0.5207 ± 0.0134 | 0.4974 ± 0.0107 | 0.5059 ± 0.0119 |
| without delta | 106 | 0.6038 ± 0.0076 | 0.5802 ± 0.0053 | 0.5888 ± 0.0038 |
| without delta2 | 106 | 0.6111 ± 0.0056 | 0.5798 ± 0.0209 | 0.5854 ± 0.0240 |
| without spectral | 134 | 0.6441 ± 0.0024 | 0.6052 ± 0.0111 | 0.6131 ± 0.0115 |
| without contrast | 132 | 0.6390 ± 0.0107 | 0.5952 ± 0.0129 | 0.6005 ± 0.0115 |

(The ablation uses its own copy of the training loop with DataLoader shuffling, so its "all features" row differs slightly from the five src/train.py runs above.)
