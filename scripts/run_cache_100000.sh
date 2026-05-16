#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/opt/miniconda3/bin/python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --work-dir data/montecarlo/work_100000 \
  --cache-out data/montecarlo/cache_100000 \
  --sample inclusive \
  --max-events 100000 \
  --chunk-size 1000
