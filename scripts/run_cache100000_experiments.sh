#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Train DFEI cache_100000 -> expected DFEI version_1 ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_DFEI.yaml

echo "=== Train IFT cache_100000 truth-replacement -> expected IFT version_2 ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_pvDFEI1_truth.yaml

echo "=== Evaluate IFT truth-replacement version_2 with explicit suffix ==="
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config config_files/cache_100000/evaluate_IFT_truth_v2.yaml

echo "=== Train IFT cache_100000 DFEI-chain -> expected IFT version_3 ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_DFEIchain_v1.yaml

echo "=== Evaluate IFT DFEI-chain version_3 at 0.5/0.5 ==="
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config config_files/cache_100000/evaluate_IFT_DFEIchain_v3_thr05.yaml

echo "=== Audit final logs ==="
/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 120 \
  --max-graphs 10000
