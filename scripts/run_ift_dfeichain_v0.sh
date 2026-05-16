#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_50000/train_IFT_DFEIchain_v0.yaml

/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config config_files/cache_50000/evaluate_IFT_DFEIchain_v1.yaml
