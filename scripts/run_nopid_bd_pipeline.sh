#!/usr/bin/env bash
# End-to-end de-contaminated (use_pid:None) pipeline for mc24 Bd->J/psiK*.
# Removes the dominant flavor-tag leakage (true MC PID) and selects the reco
# threshold on the VALIDATION split, reporting the final tagging power on TEST.
#
# MODE=smoke  -> tiny: 10k cache, 2 epochs, 10 files (validate the chain runs)
# MODE=full   -> 100k cache, 100 epochs, 100 files (overnight, real result)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/opt/miniconda3/bin/python
MODE="${MODE:-full}"
# PID input: "true" = real LHCb detector PID (already in the data) = realistic headline.
# "None" = conservative no-PID floor. "realistic" only valid for pythia (sampled response).
USE_PID="${USE_PID:-true}"
echo "USE_PID=$USE_PID"

# Channel select: bd = Bd->J/psiK* (flavour-specific, leak stress-test); bs = Bs->J/psiphi (CP, the FT benchmark)
CHANNEL="${CHANNEL:-bd}"
if [ "$CHANNEL" = "bs" ]; then
  CH_GLOB='mc24_bs_jpsiphi*.root'; CH_SAMPLE=Bs_Jpsiphi; SAMPLE="mc24_Bs_Jpsiphi"; CH_TAG=bs
else
  CH_GLOB='mc24_bd_jpsikstar*.root'; CH_SAMPLE=Bd_JpsiKst; SAMPLE="mc24_Bd_JpsiKst"; CH_TAG=bd
fi
echo "CHANNEL=$CHANNEL  SAMPLE=$SAMPLE"

# Output/label tag base. The experiment suite sets EXP_TAG per (experiment,channel,seed)
# so val-grid + final FT files + the VALSEL file are self-describing and don't collide.
EXP_TAG="${EXP_TAG:-nopid_${CH_TAG}}"
echo "EXP_TAG=$EXP_TAG"

if [ "$MODE" = "smoke" ]; then
  MAXEV=10000; EPOCHS=2; NFILES=10
  MC24_CACHE=data/montecarlo/cache_mc24_${CH_TAG}_nopid_smoke
  WORK=data/montecarlo/work_mc24_${CH_TAG}_nopid_smoke
  # A 2-epoch DFEI is undertrained -> low node/edge scores. The IFT data loader
  # runs that DFEI for PV association and prunes at this threshold; at 0.9 it would
  # delete ~all tracks and leave 0 training graphs. Smoke only validates plumbing,
  # so use a permissive threshold to keep graphs alive.
  PVTHR=0.05
else
  MAXEV="${MAXEV:-100000}"; EPOCHS=100
  # NFILES = how many cache files to load (1000 evts/file). CACHE_SUFFIX selects the
  # cache/work dir: default nopid_100k; D experiments override (e.g. 200k, tracks) with
  # SKIP_CACHE=1 to reuse a prebuilt cache. 200k needs NFILES=200 to use all events.
  NFILES="${NFILES:-100}"
  CACHE_SUFFIX="${CACHE_SUFFIX:-nopid_100k}"
  MC24_CACHE=data/montecarlo/cache_mc24_${CH_TAG}_${CACHE_SUFFIX}
  WORK=data/montecarlo/work_mc24_${CH_TAG}_${CACHE_SUFFIX}
  PVTHR=""   # keep the trained-model threshold from the template config (0.9)
fi

DFEI_DATA="data/montecarlo/cache_100000/DFEI_LHCb_LHCbcollision_normed_pt_data/"
MC24_DATA="${MC24_CACHE}/DFEI_LHCb_LHCbcollision_normed_pt_data/"
TS="$(date +%Y%m%d_%H%M%S)"

echo "############ MODE=$MODE  maxev=$MAXEV epochs=$EPOCHS nfiles=$NFILES ############"

latest_version() {
  find "LHCb_logs/$1" -maxdepth 1 -type d -name 'version_*' -printf '%f\n' 2>/dev/null \
    | sed 's/version_//' | sort -n | tail -1
}

echo "===== [1/5] Re-cache $SAMPLE (true-pid path; adds feature_names + asso_hh) ====="
if [ "${SKIP_CACHE:-0}" = "1" ] && ls "${MC24_DATA}/${SAMPLE}"/trn_data_* >/dev/null 2>&1; then
  echo "SKIP_CACHE=1 and cache present -> skipping re-cache"
else
  $PY scripts/cache_montecarlo.py \
    --input-dir data/montecarlo/root \
    --input-glob "$CH_GLOB" \
    --work-dir "$WORK" \
    --cache-out "$MC24_CACHE" \
    --sample "$CH_SAMPLE" \
    --identifier mc24 \
    --max-events "$MAXEV" \
    --chunk-size 1000 \
    --overwrite
fi

echo "===== verify re-cache produced feature_names + asso_hh ====="
$PY - "$MC24_DATA" "$SAMPLE" <<'PY'
import io, glob, sys, torch, zstandard
data, sample = sys.argv[1:3]
fs = sorted(glob.glob(f"{data}/{sample}/trn_data_*"))
assert fs, f"no trn cache files in {data}/{sample}"
g = torch.load(io.BytesIO(zstandard.ZstdDecompressor().stream_reader(open(fs[0],"rb")).read()), weights_only=False)[0]
t = g["tracks"]
print("feature_names:", getattr(t,"feature_names",None))
print("has asso_hh:", hasattr(t,"asso_hh"), "| has pid:", hasattr(t,"pid"))
assert getattr(t,"feature_names",None) is not None, "feature_names missing -> masking will fail"
assert hasattr(t,"asso_hh"), "asso_hh missing -> IFT signal masking will fail"
print("OK: cache is masking-compatible")
PY

if [ -n "${DFEI_VERSION:-}" ]; then
  echo "===== [2/5] SKIP DFEI training, reuse DFEI version_${DFEI_VERSION} ====="
  VD="$DFEI_VERSION"
else
  echo "===== [2/5] Train DFEI (use_pid:$USE_PID) on cache_100000 ====="
  DFEI_CFG="config_files/_auto_nopid_dfei.yaml"
  $PY - "$DFEI_DATA" "$EPOCHS" "$NFILES" "$DFEI_CFG" "$USE_PID" <<'PY'
import sys, yaml
data, epochs, nfiles, dst, use_pid = sys.argv[1:6]
cfg = yaml.safe_load(open("config_files/cache_100000/train_DFEI.yaml"))
cfg["settings"]["data_dir"] = data
cfg["settings"]["epochs"] = int(epochs)
cfg["settings"]["nfiles"] = [int(nfiles)]
cfg["evaluate"]["nfiles"] = [int(nfiles)]
cfg["DFEI"]["use_pid"] = use_pid
# trainer.py auto-runs a test phase after fit; DFEI's PV-missasso plots access a
# 'true_pv' column absent in this cache -> KeyError. We only need the DFEI weights
# for the IFT chain, so disable the eval plotting (model + checkpoint unaffected).
cfg["inference"]["plt_nodes"] = False
cfg["inference"]["plt_edges"] = False
cfg["inference"]["plt_pvs"] = False
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("wrote", dst)
PY
  $PY wmpgnn/analysis/trainer.py --config "$DFEI_CFG"
  VD="$(latest_version DFEI)"
fi
echo "DFEI_VERSION=$VD"

echo "===== [3/5] Train IFT DFEI-chain (use_pid:$USE_PID) on $SAMPLE ====="
IFT_CFG="config_files/_auto_nopid_ift_bd.yaml"
$PY - "$MC24_DATA" "$EPOCHS" "$NFILES" "$VD" "$SAMPLE" "$IFT_CFG" "${PVTHR:-}" "$USE_PID" "${MASK_KIN:-false}" <<'PY'
import sys, yaml
data, epochs, nfiles, vd, sample, dst, pvthr, use_pid, mask_kin = sys.argv[1:10]
cfg = yaml.safe_load(open("config_files/cache_100000/train_IFT_DFEIchain_v1.yaml"))
s = cfg["settings"]
s["data_dir"] = data
s["sample"] = [sample]
s["nfiles"] = [int(nfiles)]
s["epochs"] = int(epochs)
s["pv_model"] = int(vd)
s["mask_signal_inputs"] = True
s["mask_signal_kinematics"] = (mask_kin.lower() == "true")
s["exclude_signal_tracks"] = (__import__('os').environ.get("EXCLUDE_SIG", "false").lower() == "true")
cfg["evaluate"]["sample"] = [sample]
cfg["evaluate"]["nfiles"] = [int(nfiles)]
cfg["evaluate"]["over_write"] = ""
cfg["IFT"]["dfei_model"] = int(vd)
cfg["IFT"]["use_pid"] = use_pid
# Experiment knob: edge-to-node aggregation. HeteroNodeBlock reads it from
# GNblocks.MLP_forward (NOT the sibling GNblocks.aggregation, which is dead), so
# inject there. Empty -> leave template default (sum).
import os as _os
_agg = _os.environ.get("IFT_AGG", "").strip()
if _agg:
    cfg["IFT"]["GNblocks"]["MLP_forward"]["aggregation"] = _agg
# soft LCA: feed the DFEI's full softmax over track-track topology classes to the
# IFT edge input instead of the hard argmax class.
_lca = _os.environ.get("IFT_LCA", "").strip()
if _lca:
    cfg["IFT"]["lca_input"] = _lca
# message-passing depth: more GN blocks -> tag info propagates further across the event graph.
_nb = _os.environ.get("IFT_NBLOCKS", "").strip()
if _nb:
    cfg["IFT"]["GNblocks"]["nBlocks"] = int(_nb)
# loss variant knob (ce default / focal / dsquared / allclass); consumed by ift_lightning_module.
_loss = _os.environ.get("IFT_LOSS", "").strip()
if _loss:
    cfg["IFT"]["loss"] = _loss
if pvthr:  # smoke override so an undertrained DFEI doesn't prune all tracks
    cfg["inference"]["node_prune_thr"] = float(pvthr)
    cfg["inference"]["edge_prune_thr"] = float(pvthr)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("wrote", dst)
PY
$PY wmpgnn/analysis/trainer.py --config "$IFT_CFG"
VI="$(latest_version IFT)"
echo "IFT_VERSION=$VI"

# Helper: write an eval config (replicates the proven chain eval structure + split)
write_eval() {  # dst ver dfei split edge node over
  $PY - "$@" "$MC24_DATA" "$SAMPLE" "${MASK_KIN:-false}" <<'PY'
import sys, yaml, os
dst, ver, dfei, split, edge, node, over, data, sample, mask_kin = sys.argv[1:11]
cfg = {
  "settings": {"model_arch": "IFT", "model": int(ver), "dfei_model": int(dfei),
               "data_dir": data, "ncpu": 2, "calibration": False,
               # eval masking MUST match training, else signal info leaks back at eval time
               "mask_signal_inputs": True,
               "mask_signal_kinematics": (mask_kin.lower() == "true"),
               "exclude_signal_tracks": (os.environ.get("EXCLUDE_SIG", "false").lower() == "true")},
  "inference": {"edge_prune_thr": float(edge), "node_prune_thr": float(node)},
  "evaluate": {"sample": [sample], "nfiles": [int(os.environ.get('NFILES','100'))],
               "over_write": over, "split": split},
}
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
}

echo "===== [4/5] VALIDATION-split threshold grid (honest selection) ====="
export NFILES
# Full: real val grid. Smoke: single permissive point so the undertrained DFEI's
# PV association doesn't empty the eval graphs (plumbing validation only).
if [ "$MODE" = "smoke" ]; then GRID="0.05:0.05"; else GRID="0.4:0.5 0.5:0.5 0.5:0.4 0.6:0.5 0.9:0.9"; fi

VALSEL="outputs/_valsel_${EXP_TAG}.txt"; : > "$VALSEL"
gi=0
for en in $GRID; do
  e="${en%%:*}"; n="${en##*:}"
  gi=$((gi+1)); tag="${EXP_TAG}_val_${gi}"
  cfg="config_files/_auto_eval_${tag}.yaml"
  write_eval "$cfg" "$VI" "$VD" "val" "$e" "$n" "$tag"
  echo "--- val eval e=$e n=$n (tag=$tag) ---"
  $PY wmpgnn/analysis/evaluate.py --config "$cfg" || echo "  (eval failed for e=$e n=$n)"
  ft="LHCb_logs/IFT/version_${VI}/info_${SAMPLE}__${tag}_FT.txt"
  # Record BOTH the flat TP (per surviving candidate) and the full-chain TP
  # (flat * surviving_rows / input_events = efficiency folded in). Selection uses
  # full-chain so it does NOT reward aggressive pruning that discards hard events.
  read tp fc < <($PY - "$ft" <<'PY'
import sys, re
try:
    t = open(sys.argv[1], errors="ignore").read()
    m = re.search(r"Tagging power\s+:\s+\(([-0-9.]+)", t)
    rows = re.search(r"Full metric rows:\s*([0-9]+)", t)
    evts = re.search(r"Input events\s*:\s*([0-9]+)", t)
    tpf = float(m.group(1)) if m else float("nan")
    r = int(rows.group(1)) if rows else 0
    ev = int(evts.group(1)) if evts else 0
    fcv = (tpf * r / ev) if (ev > 0 and tpf == tpf) else float("nan")
    print(f"{tpf} {fcv}")
except FileNotFoundError:
    print("nan nan")
PY
)
  echo "$e $n $tp $fc" >> "$VALSEL"
  echo "  recorded: e=$e n=$n flat_tp=$tp full_chain_tp=$fc"
done

echo "===== pick best threshold on VALIDATION full-chain tagging power ====="
read BEST_E BEST_N < <($PY - "$VALSEL" <<'PY'
import sys
best = (-1.0, None, None)
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) != 4:
        continue
    e, n, tp, fc = p
    try:
        v = float(fc)
    except ValueError:
        continue
    if v != v:  # nan
        continue
    if v > best[0]:
        best = (v, e, n)
e, n = (best[1], best[2]) if best[1] is not None else ("0.4", "0.5")
print(e, n)
sys.stderr.write(f"BEST val full_chain_tp={best[0]} at e={e} n={n}\n")
PY
)
echo "Selected thresholds: edge=$BEST_E node=$BEST_N"

echo "===== [5/5] FINAL evaluation on TEST split at selected threshold ====="
final_tag="${EXP_TAG}_test_final"
final_cfg="config_files/_auto_eval_${final_tag}.yaml"
write_eval "$final_cfg" "$VI" "$VD" "tst" "$BEST_E" "$BEST_N" "$final_tag"
$PY wmpgnn/analysis/evaluate.py --config "$final_cfg"

echo "############ DONE ############"
echo "DFEI_VERSION=$VD  IFT_VERSION=$VI  selected edge=$BEST_E node=$BEST_N"
echo "Validation grid (edge node tagging_power):"; cat "$VALSEL"
echo "Final TEST tagging power:"
grep -H "Tagging power" "LHCb_logs/IFT/version_${VI}/info_${SAMPLE}__${final_tag}_FT.txt" 2>/dev/null || echo "  (final FT file not found)"
