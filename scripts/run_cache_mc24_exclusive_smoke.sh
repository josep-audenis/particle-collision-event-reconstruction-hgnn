#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs

echo "=== Smoke cache MC24 Bd_JpsiKst ==="
/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --input-glob 'mc24_bd_jpsikstar*.root' \
  --work-dir data/montecarlo/work_mc24_smoke \
  --cache-out data/montecarlo/cache_mc24_smoke \
  --sample Bd_JpsiKst \
  --identifier mc24 \
  --max-events 10000 \
  --chunk-size 1000 \
  --overwrite

echo "=== Smoke cache MC24 Bs_Jpsiphi ==="
/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --input-glob mc24_bs_jpsiphi.root \
  --work-dir data/montecarlo/work_mc24_smoke \
  --cache-out data/montecarlo/cache_mc24_smoke \
  --sample Bs_Jpsiphi \
  --identifier mc24 \
  --max-events 5000 \
  --chunk-size 1000 \
  --overwrite
