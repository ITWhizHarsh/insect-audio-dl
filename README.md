# Deep Learning for Acoustic Insect Species Identification

ICT 4442 Deep Learning mini project, School of Computer Engineering, MIT Manipal.

A comparison of four architecture families for identifying insect species
(Orthoptera and Cicadidae) from short audio recordings, under one shared
evaluation protocol, on [InsectSet459](https://doi.org/10.5281/zenodo.14056457)
(Faiß, Ghani & Stowell, *Scientific Data* 13, 499, 2026).

| Model | Family | Input | Owner |
|---|---|---|---|
| MLP | classical baseline | 146-d handcrafted descriptors | Abhishek Patro |
| VGG-style CNN (+ EfficientNetV2-S, final phase) | CNN | 128-band log-mel | Saurabh Tiwari |
| CRNN (CNN + BiLSTM + attention pooling) | recurrent / hybrid | 128-band log-mel | Saksham Gupta |
| Audio Spectrogram Transformer | attention | log-mel patches | Harsh Kumar Roy |

## Status (interim, 2 Oct 2026)

- Data acquisition by HTTP range requests (no need to download the 84 GB archives)
- Preprocessing: decode, mono, resample to 44.1 kHz, peak-normalise, 5 s chunks
- Shared features: handcrafted descriptors and log-mel spectrograms, cached once
- Shared evaluation: file-level macro-F1 (primary), accuracy, macro P/R, chunk-level scores
- MLP and CNN trained on a 12-species pilot subset; CRNN implemented
- Metadata analysis: long-tail distribution and temperature coverage

## Layout

```
src/
  config.py             common protocol: sample rate, chunking, spectrogram settings, seed
  data/fetch_subset.py  pull selected recordings out of the remote Zenodo zips
  data/preprocess.py    load / resample / normalise / chunk
  data/verify_splits.py leakage and coverage checks on the official split
  features/             handcrafted.py (MLP), logmel.py (CNN/CRNN/AST), augment.py
  models/               mlp.py, cnn.py, crnn.py
  utils/imbalance.py    class weights and balanced sampling for the long tail
  eval/metrics.py       file-level aggregation, metrics, confusion matrices
  eval/compare.py       single comparison table across models
  train.py              shared training loop and model selection
scripts/build_features.py  run preprocessing once and cache inputs
analysis/eda_metadata.py   long-tail and temperature analysis of the metadata
results/                   metrics (json/csv) and figures; data and checkpoints are not committed
```

## Reproduce the pilot

```bash
pip install -r requirements.txt
python -m src.data.fetch_subset --n-species 12 --out data/raw/pilot     # ~0.3 GB
python scripts/build_features.py --manifest data/raw/pilot/manifest.csv --name pilot
python -m src.train --model mlp --data pilot --epochs 150 --batch 64 --wd 1e-4
python -m src.train --model cnn --data pilot --epochs 40 --augment
python -m src.eval.compare --data pilot
python analysis/eda_metadata.py
```

## Data licence

Recordings are CC-BY-4.0 / CC-BY-NC-4.0 / CC0 as listed per file in the
annotation CSV. Audio is never committed to this repository.

## Use of AI tools

Code and report drafts were produced with the help of an LLM assistant
(Claude, Anthropic) under the team's direction; every member reviewed and is
responsible for the part they own. Details are given in the report.
