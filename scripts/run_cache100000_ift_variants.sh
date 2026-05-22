#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs

latest_version() {
  local model="$1"
  find "LHCb_logs/${model}" -maxdepth 1 -type d -name 'version_*' -printf '%f\n' 2>/dev/null \
    | sed 's/version_//' | sort -n | tail -1
}

write_eval_yaml() {
  local dst="$1"
  local ift_version="$2"
  local overwrite="$3"
  /opt/miniconda3/bin/python - "$dst" "$ift_version" "$overwrite" <<'PY'
from pathlib import Path
import sys
import yaml

dst, ift_version, overwrite = sys.argv[1:4]
cfg = {
    "settings": {
        "model_arch": "IFT",
        "model": int(ift_version),
        "dfei_model": 1,
        "data_dir": "data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data/",
        "ncpu": 2,
        "calibration": False,
    },
    "inference": {
        "edge_prune_thr": 0.4,
        "node_prune_thr": 0.5,
    },
    "evaluate": {
        "sample": ["00353851_inclusive"],
        "nfiles": [100],
        "over_write": overwrite,
    },
}
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

run_eval() {
  local ift_version="$1"
  local overwrite="$2"
  local cfg="config_files/cache_100000/evaluate_IFT_${overwrite}.yaml"
  write_eval_yaml "$cfg" "$ift_version" "$overwrite"
  /opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py --config "$cfg"
}

echo "=== Backfill 100k truth/PV control report with normalized metrics ==="
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config config_files/cache_100000/evaluate_IFT_truth_v2.yaml

echo "=== Backfill 100k DFEI-chain best report with normalized metrics ==="
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e04_n05.yaml

echo "=== Train 100k DFEI-chain attention IFT ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_DFEIchain_attention.yaml
attention_version="$(latest_version IFT)"
echo "IFT_ATTENTION_VERSION=$attention_version"

echo "=== Evaluate 100k DFEI-chain attention IFT at edge=0.4 node=0.5 ==="
run_eval "$attention_version" "cache100000_dfeichain_attention_e04_n05"

echo "=== Train 100k DFEI-chain mean aggregation IFT ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_DFEIchain_mean.yaml
mean_version="$(latest_version IFT)"
echo "IFT_MEAN_VERSION=$mean_version"

echo "=== Evaluate 100k DFEI-chain mean aggregation IFT at edge=0.4 node=0.5 ==="
run_eval "$mean_version" "cache100000_dfeichain_mean_e04_n05"

echo "=== Train 100k DFEI-chain soft-LCA IFT ==="
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config config_files/cache_100000/train_IFT_DFEIchain_soft_lca.yaml
soft_lca_version="$(latest_version IFT)"
echo "IFT_SOFT_LCA_VERSION=$soft_lca_version"

echo "=== Evaluate 100k DFEI-chain soft-LCA IFT at edge=0.4 node=0.5 ==="
run_eval "$soft_lca_version" "cache100000_dfeichain_soft_lca_e04_n05"

echo "=== Known-result parser smoke ==="
/opt/miniconda3/bin/python scripts/compare_all_results.py || true
/opt/miniconda3/bin/python scripts/compare_ift_reports.py || true

echo "=== Finished 100k IFT variants ==="
