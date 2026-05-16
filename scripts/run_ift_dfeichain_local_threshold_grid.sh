#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for config in \
  config_files/cache_50000/evaluate_IFT_DFEIchain_v1_e05_n04.yaml \
  config_files/cache_50000/evaluate_IFT_DFEIchain_v1_e04_n05.yaml \
  config_files/cache_50000/evaluate_IFT_DFEIchain_v1_e06_n05.yaml \
  config_files/cache_50000/evaluate_IFT_DFEIchain_v1_e05_n06.yaml
do
  echo "=== $config ==="
  /opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py --config "$config"
done
