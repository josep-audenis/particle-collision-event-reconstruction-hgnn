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

Additional experiment configs live in `examples`.

Before running, update each config path under `settings.data_dir`, plus any
checkpoint or log path needed by your environment.

## Training

Run from `config_files`:

```bash
cd config_files
python ../wmpgnn/analysis/trainer.py --config train_DFEI.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT.yaml
```

Run example sweeps from `examples`:

```bash
cd examples
bash run.sh
```

or use the summary wrapper:

```bash
cd examples
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
