#!/usr/bin/env bash
# Track-type ablation: isolate the marginal contribution of downstream vs upstream
# tracks on top of long tracks. Completes the 2x2 with the existing baseline (long)
# and tracks_all (long+down+up) points, all at the 100k baseline config.
#
#   tracks_ldown = long,down   (cache suffix: ldown)
#   tracks_lup   = long,up     (cache suffix: lup)
#
# Phase 1: build the 4 caches (long+down / long+up, bd/bs), dropping each WORK dir
#          after build to keep disk in check (server is at ~96%).
# Phase 2: 2 exp x 2 ch x 3 seeds = 12 runs, mirroring run_experiment_suite.sh
#          (GPU pick, orphan reap, suite_record, DONE markers). SKIP_CACHE=1.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/opt/miniconda3/bin/python
SUITE=outputs/suite
RESULTS="$SUITE/results_tracktype.tsv"
mkdir -p "$SUITE"

SEEDS="${SEEDS:-42 43 44}"
CHANNELS="${CHANNELS:-bd bs}"
GPU_ALLOWED="${GPU_ALLOWED:-1}"
GPU_MEM_FREE_MIB="${GPU_MEM_FREE_MIB:-1500}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"   # abort a cache build if free disk below this

# exp | track_types | cache_suffix
EXPERIMENTS=(
  "tracks_ldown|long,down|ldown"
  "tracks_lup|long,up|lup"
)

ch_glob()   { [ "$1" = bs ] && echo 'mc24_bs_jpsiphi*.root' || echo 'mc24_bd_jpsikstar*.root'; }
ch_sample() { [ "$1" = bs ] && echo 'Bs_Jpsiphi' || echo 'Bd_JpsiKst'; }

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

pick_gpu() {
  while true; do
    local idx used
    while IFS=',' read -r idx used; do
      idx="$(echo "$idx" | tr -d ' ')"; used="$(echo "$used" | tr -d ' ')"
      echo " $GPU_ALLOWED " | grep -q " $idx " || continue
      if [ "$used" -lt "$GPU_MEM_FREE_MIB" ]; then echo "$idx"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "[$(date '+%F %T')] all GPUs busy, waiting..." >&2; sleep 60
  done
}

echo "############ TRACK-TYPE ABLATION start $(date) ############"

# ---- Phase 1: build caches ----
for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r exp ttypes suffix <<< "$entry"
  for ch in $CHANNELS; do
    cache="data/montecarlo/cache_mc24_${ch}_${suffix}"
    work="data/montecarlo/work_mc24_${ch}_${suffix}"
    # cache subdir is prefixed by --identifier (mc24) + sample, e.g. mc24_Bd_JpsiKst
    data="${cache}/DFEI_LHCb_LHCbcollision_normed_pt_data/mc24_$(ch_sample "$ch")"
    if ls "$data"/trn_data_* >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] cache exists, skip build: $cache"; continue
    fi
    fg="$(free_gb)"
    if [ "$fg" -lt "$MIN_FREE_GB" ]; then
      echo "[$(date '+%F %T')] ABORT: only ${fg}G free (<${MIN_FREE_GB}G) before building $cache" >&2
      exit 3
    fi
    echo "[$(date '+%F %T')] BUILD cache $cache  TRACK_TYPES=$ttypes  (free ${fg}G)"
    TRACK_TYPES="$ttypes" $PY scripts/cache_montecarlo.py \
      --input-dir data/montecarlo/root \
      --input-glob "$(ch_glob "$ch")" \
      --work-dir "$work" \
      --cache-out "$cache" \
      --sample "$(ch_sample "$ch")" \
      --identifier mc24 \
      --max-events 100000 \
      --chunk-size 1000 \
      --overwrite
    rc=$?
    if [ "$rc" -ne 0 ] || ! ls "$data"/trn_data_* >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] CACHE BUILD FAILED for $cache (rc=$rc)" >&2; exit 4
    fi
    # drop the build intermediate to reclaim disk (training reads only $cache)
    echo "[$(date '+%F %T')] built OK; removing WORK intermediate $work"
    rm -rf "$work"
  done
done

# ---- Phase 2: train + eval ----
for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r exp ttypes suffix <<< "$entry"
  for ch in $CHANNELS; do
    for seed in $SEEDS; do
      tag="${exp}_${ch}_s${seed}"
      [ -f "$SUITE/DONE_${tag}" ] && { echo "[$(date '+%F %T')] SKIP $tag (DONE)"; continue; }
      gpu="$(pick_gpu)"
      log="$SUITE/${tag}.log"
      echo "=========================================================="
      echo "[$(date '+%F %T')] RUN $tag  GPU=$gpu  CACHE_SUFFIX=$suffix"
      timeout "$RUN_TIMEOUT" env \
        CUDA_VISIBLE_DEVICES="$gpu" TFG_SEED="$seed" EXP_TAG="$tag" \
        SKIP_CACHE=1 DFEI_VERSION=5 USE_PID=true CHANNEL="$ch" MODE=full \
        CACHE_SUFFIX="$suffix" NFILES=100 \
        bash scripts/run_nopid_bd_pipeline.sh > "$log" 2>&1
      rc=$?
      pkill -KILL -f _auto_nopid_ift_bd 2>/dev/null || true
      pkill -KILL -f wmpgnn/analysis/trainer.py 2>/dev/null || true
      sleep 5
      if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc (see $log)"
        echo -e "$(date '+%F %T')\t$tag\trc=$rc" >> "$SUITE/failures_tracktype.txt"; continue
      fi
      if $PY scripts/suite_record.py --results "$RESULTS" \
           --exp "$exp" --channel "$ch" --seed "$seed" --log "$log"; then
        touch "$SUITE/DONE_${tag}"
        echo "[$(date '+%F %T')] OK $tag recorded"
      else
        echo "[$(date '+%F %T')] WARN $tag record failed (see $log)"
      fi
    done
  done
done

echo "############ ABLATION done $(date) ############"
[ -f "$RESULTS" ] && cat "$RESULTS"
