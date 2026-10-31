# Deep Learning for Acoustic Insect Species Identification

ICT 4442 Deep Learning mini project, School of Computer Engineering, MIT Manipal.

A comparison of four architecture families for identifying insect species
(Orthoptera and Cicadidae) from short audio recordings, under one shared
evaluation protocol, on [InsectSet459](https://doi.org/10.5281/zenodo.14056457)
(Faiß, Ghani & Stowell, *Scientific Data* 13, 499, 2026).

| Model | Family | Input | Owner |
|---|---|---|---|
| MLP | classical baseline | 146-d handcrafted descriptors | Abhishek Patro |
| VGG-style CNN, EfficientNetV2-S (ImageNet) | CNN | 128-band log-mel | Saurabh Tiwari |
| CRNN (CNN + BiLSTM + attention pooling) | recurrent / hybrid | 128-band log-mel | Saksham Gupta |
| Audio Spectrogram Transformer (AudioSet) | attention | log-mel patches | Harsh Kumar Roy |

## Status

**Interim (2 Oct 2026).** Range-request data acquisition, shared preprocessing and
evaluation, MLP / CNN / CRNN on a 12-species pilot, metadata analysis.

**Final phase (from 14 Oct 2026).**

- 40-species benchmark subset (20 Orthoptera + 20 Cicadidae, 1,991 recordings,
  official split) cached locally; all models train on the same cached inputs
- All four families implemented: MLP, VGG-style CNN and EfficientNetV2-S (ImageNet),
  CRNN (LSTM / GRU), Audio Spectrogram Transformer (AudioSet weights, our own
  PyTorch implementation, checked against the reference implementation)
- Common protocol extended: multi-seed runs, saved file-level predictions,
  recordist-disjoint test scores, bootstrap confidence intervals and paired tests
- Controlled experiments: augmentation ablation (CNN), recurrent cell (CRNN),
  class-imbalance strategies on a 10:1 long-tailed training set, handcrafted
  feature-group ablation and classical baselines (MLP), temperature conditioning
  (Orthoptera with recorded temperature)
- Experiment plans in `experiments/`; the MLP, CNN and CRNN run on the laptop,
  EfficientNetV2-S and AST on a free GPU via `notebooks/colab_gpu_training.ipynb`

## Results (40-species benchmark, test set, file-level macro-F1)

| Model | Params | Seeds | Macro-F1 (mean ± sd) | Recordist-unseen (seed ensemble) |
|---|---|---|---|---|
| EfficientNetV2-S (ImageNet) | 20.2 M | 1 | 0.847 | 0.791 |
| AST (AudioSet, laptop setting) | 85.4 M | 3 | 0.831 ± 0.015 | 0.808 |
| CNN (VGG-style, from scratch) | 1.19 M | 3 | 0.797 ± 0.013 | 0.734 |
| CRNN (BiGRU variant) | 1.41 M | 1 | 0.718 | 0.655 |
| CRNN (BiLSTM) | 1.61 M | 3 | 0.673 ± 0.013 | 0.659 |
| MLP (146 handcrafted features) | 0.22 M | 5 | 0.609 ± 0.007 | 0.545 |
| AST from random initialisation | 85.4 M | 1 | 0.409 | 0.299 |

The two pretrained models are statistically tied (paired bootstrap p = 0.72) and both
ahead of the CNN trained from scratch (EfficientNetV2-S p = 0.01, AST p = 0.10); every other
gap between families is significant. Recorded temperature (FiLM conditioning, 36 Orthoptera
species) adds no information over shuffled temperatures, and on a 10:1 long-tailed training
set logit adjustment is the only imbalance strategy that helps (+0.051, p = 0.013).
Full tables: `results/comparison_species40.md`, `results/metrics/model_comparison_species40_pairs.csv`,
`results/metrics/longtail_species40_lt10.csv`, `results/metrics/temperature_orthotemp.csv`.

## Layout

```
src/
  config.py             common protocol: sample rate, chunking, spectrogram settings, seed
  data/fetch_subset.py  pull selected recordings out of the remote Zenodo zips
  data/preprocess.py    load / resample / normalise / chunk
  data/verify_splits.py leakage and coverage checks on the official split
  data/recordist.py     metadata join and recordist-disjoint test subset
  features/             handcrafted.py (MLP), logmel.py (CNN/CRNN/AST), augment.py
  models/               mlp.py, cnn.py, effnet.py, crnn.py, ast.py, conditioning.py
  utils/imbalance.py    class weights, balanced sampling, long-tail subsets, logit adjustment
  eval/metrics.py       file-level aggregation, metrics, confusion matrices
  eval/stats.py         bootstrap CIs, paired bootstrap, McNemar, seed aggregation
  eval/compare.py       one comparison table across models (mean +- sd over seeds)
  train.py              shared training loop and model selection
scripts/build_features.py  run preprocessing once and cache inputs
scripts/run_queue.py       run an experiment plan, skipping finished runs
experiments/               experiment plans (one src.train command per line)
analysis/                  metadata EDA, audits, error / long-tail analysis, MLP ablations
notebooks/                 Colab notebooks (feature streaming, GPU training)
results/                   metrics, per-file predictions and figures; audio and checkpoints are not committed
```

## Reproduce

```bash
pip install -r requirements.txt

# 40-species benchmark subset (~1.4 GB of audio, official split)
python -m src.data.fetch_subset --from-manifest data/meta/species40_manifest.csv --out data/raw/species40 --workers 12
python scripts/build_features.py --manifest data/raw/species40/manifest.csv --name species40

python scripts/run_queue.py experiments/species40_main.txt        # MLP, CNN, CRNN (laptop)
python scripts/run_queue.py experiments/species40_ablations.txt   # augmentation, GRU, long tail
python scripts/run_queue.py experiments/species40_gpu.txt         # EfficientNetV2-S, AST (GPU)
python -m src.eval.compare --data species40 --markdown
python -m pytest tests -q
```

The 12-species pilot of the interim report is reproduced with
`python -m src.data.fetch_subset --n-species 12 --out data/raw/pilot` and the same scripts
with `--data pilot`.

## Data licence

Recordings are CC-BY-4.0 / CC-BY-NC-4.0 / CC0 as listed per file in the
annotation CSV. Audio is never committed to this repository.

## Use of AI tools

Code was produced with the help of an LLM assistant
(Claude, Anthropic) under the team's direction; every member reviewed and is
responsible for the part they own. Details are given in the report.
