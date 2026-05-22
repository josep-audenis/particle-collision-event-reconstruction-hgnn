#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_version="${1:-4}"
dfei_version="${2:-1}"

run_eval() {
  local edge="$1"
  local node="$2"
  local suffix="$3"
  local config="outputs/evaluate_IFT_DFEIchain_trained_v${model_version}_${suffix}.yaml"

  cat > "$config" <<EOF
settings:
  model_arch: "IFT"
  model: $model_version
  dfei_model: $dfei_version
  data_dir: "data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data/"
  ncpu: 2
  calibration: false

inference:
  edge_prune_thr: $edge
  node_prune_thr: $node

evaluate:
  sample: [ "00353851_inclusive" ]
  nfiles: [ 100 ]
  over_write: "cache100000_dfeichain_trained_${suffix}"
EOF

  echo "=== model=${model_version} edge=${edge} node=${node} suffix=${suffix} ==="
  /opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py --config "$config"
}

run_eval 0.4 0.4 e04_n04
run_eval 0.3 0.5 e03_n05
run_eval 0.4 0.5 e04_n05_repeat
run_eval 0.5 0.5 e05_n05
run_eval 0.4 0.6 e04_n06

/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 200 \
  --max-graphs 10000
