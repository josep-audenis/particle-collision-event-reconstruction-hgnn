# LHCb HGNN for DFEI and IFT

This repository contains the LHCb heterogeneous graph neural network workflow for
DFEI reconstruction and IFT flavour tagging. The current code path is the
Lightning-based implementation under `wmpgnn/analysis`, with model components in
`wmpgnn/model`, loaders in `wmpgnn/data_loader`, reconstruction helpers in
`wmpgnn/reconstruction`, and performance tools in `wmpgnn/performance`.

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

For PyTorch Geometric CUDA wheels, `requirements.txt` points to the PyG wheel
index for Torch 2.6.0 and CUDA 12.6.

## Configs

Main configs live in `config_files`:

- `train_DFEI.yaml`
- `train_IFT.yaml`
- `evaluate_DFEI.yaml`
- `evaluate_IFT.yaml`

Additional experiment configs live under `config_files/cache_10000`,
`config_files/cache_50000`, `config_files/cache_100000`, and `config_files/cache_200000`.

Before running, update each config path under `settings.data_dir`, plus any
checkpoint or log path needed by your environment.

## Cached Dataset Creation

Training configs expect cached PyG graph files, not raw ROOT files. For example,
`config_files/cache_50000/train_DFEI.yaml` points to:

```text
data/montecarlo/cache_50000/DFEI_LHCb_LHCbcollision_normed_pt_data/
```

That directory must contain one subdirectory per sample, with files named like:

```text
00353851_inclusive/
  trn_data_*.pt.zst
  val_data_*.pt.zst
  tst_data_*.pt.zst
  config.yaml
```

Create these cached files from raw LHCb Monte Carlo `*.mc.root` tuples with:

```bash
python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --work-dir data/montecarlo_work/cache_50000 \
  --cache-out data/montecarlo/cache_50000 \
  --sample inclusive \
  --identifier 00353851 \
  --max-events 50000
```

The script runs three stages: raw `EvtTuple/Tracks` conversion, particle/relation
ROOT construction, then compressed graph caching. It writes normalized LHCb DFEI
graphs under:

```text
data/montecarlo/cache_50000/DFEI_LHCb_LHCbcollision_normed_pt_data/00353851_inclusive/
```

After caching, verify the trainer can see files:

```bash
find data/montecarlo/cache_50000 -type f -name 'trn_data_*' | head
find data/montecarlo/cache_50000 -type f -name 'val_data_*' | head
find data/montecarlo/cache_50000 -type f -name 'tst_data_*' | head
```

If training fails with `ValueError: min() arg is an empty sequence`, the loader
found no `trn_data_*` files at `settings.data_dir/<sample>/`. Check
`settings.data_dir`, `settings.sample`, and the cache command output path.

## Training

Run from `config_files`:

```bash
cd config_files
python ../wmpgnn/analysis/trainer.py --config train_DFEI.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT.yaml
```

Run example sweeps from `scripts`:

```bash
cd scripts
bash run.sh
```

or use the summary wrapper:

```bash
cd scripts
bash fancy_run.sh
```

## Evaluation

Run from `config_files`:

```bash
cd config_files
python ../wmpgnn/analysis/evaluate.py --config evaluate_DFEI.yaml
python ../wmpgnn/analysis/evaluate.py --config evaluate_IFT.yaml
```

Outputs are written to the log/output paths configured in the YAML files.

## Dataset Notes

Dataset documentation from the imported snapshot is in `Dataset.md`. Large local
datasets, model checkpoints, and generated outputs should stay outside git.
