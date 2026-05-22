#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

before="$(mktemp)"
after="$(mktemp)"
find LHCb_logs/IFT -maxdepth 1 -type d -name 'version_*' -printf '%f\n' 2>/dev/null | sort > "$before"

/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_DFEIchain_v1.yaml

find LHCb_logs/IFT -maxdepth 1 -type d -name 'version_*' -printf '%f\n' 2>/dev/null | sort > "$after"
new_version="$(comm -13 "$before" "$after" | sed 's/version_//' | sort -n | tail -1)"

if [[ -z "$new_version" ]]; then
  new_version="$(find LHCb_logs/IFT -maxdepth 1 -type d -name 'version_*' -printf '%f\n' | sed 's/version_//' | sort -n | tail -1)"
fi

echo "NEW_IFT_VERSION=$new_version"

eval_config="outputs/evaluate_IFT_DFEIchain_trained_v${new_version}_e04_n05.yaml"
cat > "$eval_config" <<EOF
settings:
  model_arch: "IFT"
  model: $new_version
  dfei_model: 1
  data_dir: "data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data/"
  ncpu: 2
  calibration: false

inference:
  edge_prune_thr: 0.4
  node_prune_thr: 0.5

evaluate:
  sample: [ "00353851_inclusive" ]
  nfiles: [ 100 ]
  over_write: "cache100000_dfeichain_trained_e04_n05"
EOF

/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py --config "$eval_config"

/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 200 \
  --max-graphs 10000

/opt/miniconda3/bin/python scripts/compare_all_results.py
