#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs

echo "=== Cache MC24 Bs_Jpsiphi 100k sample ==="
/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --input-glob mc24_bs_jpsiphi.root \
  --work-dir data/montecarlo/work_mc24_exclusive_100k \
  --cache-out data/montecarlo/cache_mc24_exclusive_100k \
  --sample Bs_Jpsiphi \
  --identifier mc24 \
  --max-events 100000 \
  --chunk-size 1000 \
  --overwrite
