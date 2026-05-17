#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for config in \
  config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e05_n04.yaml \
  config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e04_n05.yaml \
  config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e06_n05.yaml \
  config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e05_n06.yaml
do
  echo "=== $config ==="
  /opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py --config "$config"
done

/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 200 \
  --max-graphs 10000
