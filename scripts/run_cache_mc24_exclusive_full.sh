#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs

echo "=== Cache MC24 Bd_JpsiKst full sample ==="
/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --input-glob 'mc24_bd_jpsikstar*.root' \
  --work-dir data/montecarlo/work_mc24_exclusive \
  --cache-out data/montecarlo/cache_mc24_exclusive \
  --sample Bd_JpsiKst \
  --identifier mc24 \
  --chunk-size 1000

echo "=== Cache MC24 Bs_Jpsiphi full sample ==="
/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --input-glob mc24_bs_jpsiphi.root \
  --work-dir data/montecarlo/work_mc24_exclusive \
  --cache-out data/montecarlo/cache_mc24_exclusive \
  --sample Bs_Jpsiphi \
  --identifier mc24 \
  --chunk-size 1000
