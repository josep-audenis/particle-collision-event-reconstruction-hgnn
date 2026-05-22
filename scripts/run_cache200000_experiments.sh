#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_DIR="config_files/cache_200000"
CACHE_DIR="data/montecarlo/cache_200000/DFEI_LHCb_LHCbcollision_normed_pt_data"
mkdir -p "$CONFIG_DIR" outputs

latest_version() {
  local model="$1"
  find "LHCb_logs/${model}" -maxdepth 1 -type d -name 'version_*' -printf '%f\n' 2>/dev/null \
    | sed 's/version_//' | sort -n | tail -1
}

write_yaml() {
  local src="$1"
  local dst="$2"
  local dfei_version="${3:-}"
  local ift_version="${4:-}"
  /opt/miniconda3/bin/python - "$src" "$dst" "$dfei_version" "$ift_version" <<'PY'
from pathlib import Path
import sys

src, dst, dfei_version, ift_version = sys.argv[1:5]
text = Path(src).read_text()
text = text.replace("cache_100000", "cache_200000")
text = text.replace("nfiles: [ 100 ]", "nfiles: [ 200 ]")
text = text.replace("cache100000", "cache200000")
if dfei_version:
    text = text.replace("pv_model: 1", f"pv_model: {dfei_version}")
    text = text.replace("dfei_model: 1", f"dfei_model: {dfei_version}")
if ift_version:
    text = text.replace("model: 2", f"model: {ift_version}")
    text = text.replace("model: 3", f"model: {ift_version}")
Path(dst).write_text(text)
PY
}

echo "=== Audit cache_200000 before training ==="
/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir "$CACHE_DIR" \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 240 \
  --max-graphs 10000

echo "=== Train DFEI cache_200000 ==="
write_yaml config_files/cache_100000/train_DFEI.yaml "$CONFIG_DIR/train_DFEI.yaml"
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config "$CONFIG_DIR/train_DFEI.yaml"
dfei_version="$(latest_version DFEI)"
echo "DFEI_VERSION=$dfei_version"

echo "=== Train IFT cache_200000 truth/PV control ==="
write_yaml config_files/cache_100000/train_IFT_pvDFEI1_truth.yaml "$CONFIG_DIR/train_IFT_truth.yaml" "$dfei_version"
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config "$CONFIG_DIR/train_IFT_truth.yaml"
ift_truth_version="$(latest_version IFT)"
echo "IFT_TRUTH_VERSION=$ift_truth_version"

echo "=== Evaluate IFT cache_200000 truth/PV control ==="
write_yaml config_files/cache_100000/evaluate_IFT_truth_v2.yaml "$CONFIG_DIR/evaluate_IFT_truth.yaml" "$dfei_version" "$ift_truth_version"
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config "$CONFIG_DIR/evaluate_IFT_truth.yaml"

echo "=== Train IFT cache_200000 DFEI-chain ==="
write_yaml config_files/cache_100000/train_IFT_DFEIchain_v1.yaml "$CONFIG_DIR/train_IFT_DFEIchain.yaml" "$dfei_version"
/opt/miniconda3/bin/python wmpgnn/analysis/trainer.py \
  --config "$CONFIG_DIR/train_IFT_DFEIchain.yaml"
ift_chain_version="$(latest_version IFT)"
echo "IFT_CHAIN_VERSION=$ift_chain_version"

echo "=== Evaluate IFT cache_200000 DFEI-chain tuned 0.4/0.5 ==="
write_yaml config_files/cache_100000/evaluate_IFT_DFEIchain_v3_e04_n05.yaml "$CONFIG_DIR/evaluate_IFT_DFEIchain_e04_n05.yaml" "$dfei_version" "$ift_chain_version"
/opt/miniconda3/bin/python wmpgnn/analysis/evaluate.py \
  --config "$CONFIG_DIR/evaluate_IFT_DFEIchain_e04_n05.yaml"

echo "=== Audit final logs ==="
/opt/miniconda3/bin/python scripts/audit_dataset_conversion.py \
  --cache-dir "$CACHE_DIR" \
  --log-dir LHCb_logs \
  --split tst \
  --max-files 240 \
  --max-graphs 10000

echo "=== Compare known results ==="
/opt/miniconda3/bin/python scripts/compare_all_results.py || true
