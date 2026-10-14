#!/usr/bin/env bash
# Unattended multi-seed experiment queue. Runs each experiment x channel x seed
# through run_nopid_bd_pipeline.sh (train IFT + full-chain val selection + test),
# records results, and aggregates mean+/-std. Sequential, auto-picks a free GPU,
# resumable via DONE markers, continues past a failed run.
#
#   nohup bash scripts/run_experiment_suite.sh > outputs/suite/driver.log 2>&1 &
#
# Env knobs: SEEDS (default "42 43 44"), CHANNELS (default "bd bs"),
#            EXPERIMENTS_FILTER (space list of exp names to run; default all),
#            GPU_MEM_FREE_MIB (a GPU counts as free below this; default 1500),
#            RUN_TIMEOUT (per-run wall cap; default 5h).
set -uo pipefail   # NOT -e: one failed run must not abort the queue
cd ~/tfg/hgnn
PY=/opt/miniconda3/bin/python

SUITE=outputs/suite
mkdir -p "$SUITE"
RESULTS="$SUITE/results.tsv"

SEEDS="${SEEDS:-42 43 44}"
CHANNELS="${CHANNELS:-bd bs}"
GPU_MEM_FREE_MIB="${GPU_MEM_FREE_MIB:-1500}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"   # 8h: 200k/tracks runs are ~2x baseline (~1.5h); headroom for hangs
# Which GPU indices the suite may use. Default GPU1 only: GPU0 is intermittently
# shared with other users, and a contended run overruns the timeout. Set
# GPU_ALLOWED="0 1" to use both once GPU0 is reliably free.
GPU_ALLOWED="${GPU_ALLOWED:-1}"

# Experiment table: "name|ENV_ASSIGNMENTS" (space-separated KEY=VAL passed to the
# run's env; empty = template defaults). Config-only experiments (no re-cache).
# Loss variants (need the IFT_LOSS code) and re-cache experiments (IP, track-types,
# 200k) are appended once their code / cache-build + disk handling are verified.
EXPERIMENTS=(
  "baseline|"
  "agg_mean|IFT_AGG=mean"
  "agg_attention|IFT_AGG=attention"
  "lca_soft|IFT_LCA=probabilities"
  "nblocks3|IFT_NBLOCKS=3"
  "nblocks4|IFT_NBLOCKS=4"
  "lca_soft_nb3|IFT_LCA=probabilities IFT_NBLOCKS=3"
  "loss_focal|IFT_LOSS=focal"
  "loss_allclass|IFT_LOSS=allclass"
  "data_200k|CACHE_SUFFIX=200k NFILES=200"
  "tracks_all|CACHE_SUFFIX=tracks"
  "data200k_lca_nb3|CACHE_SUFFIX=200k NFILES=200 IFT_LCA=probabilities IFT_NBLOCKS=3"
  "data200k_tracks_lca_nb3|CACHE_SUFFIX=200ktracks NFILES=200 IFT_LCA=probabilities IFT_NBLOCKS=3"
)

pick_gpu() {
  # Echo the index of a GPU whose used memory is below the free threshold.
  # Waits (60s poll) until one is available so the queue never races a busy GPU.
  while true; do
    local line idx used
    while IFS=',' read -r idx used; do
      idx="$(echo "$idx" | tr -d ' ')"; used="$(echo "$used" | tr -d ' ')"
      echo " $GPU_ALLOWED " | grep -q " $idx " || continue   # only allowed GPUs
      if [ "$used" -lt "$GPU_MEM_FREE_MIB" ]; then
        echo "$idx"; return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "[$(date '+%F %T')] all GPUs busy (>=${GPU_MEM_FREE_MIB} MiB), waiting..." >&2
    sleep 60
  done
}

echo "############ EXPERIMENT SUITE start $(date) ############"
echo "seeds=[$SEEDS] channels=[$CHANNELS] timeout=$RUN_TIMEOUT free<${GPU_MEM_FREE_MIB}MiB"

for entry in "${EXPERIMENTS[@]}"; do
  exp="${entry%%|*}"; expenv="${entry#*|}"
  if [ -n "${EXPERIMENTS_FILTER:-}" ] && ! echo " $EXPERIMENTS_FILTER " | grep -q " $exp "; then
    echo "skip experiment $exp (not in EXPERIMENTS_FILTER)"; continue
  fi
  for ch in $CHANNELS; do
    for seed in $SEEDS; do
      tag="${exp}_${ch}_s${seed}"
      if [ -f "$SUITE/DONE_${tag}" ]; then
        echo "[$(date '+%F %T')] SKIP $tag (DONE marker present)"; continue
      fi
      gpu="$(pick_gpu)"
      log="$SUITE/${tag}.log"
      echo "=========================================================="
      echo "[$(date '+%F %T')] RUN $tag  GPU=$gpu  env='${expenv:-defaults}'"
      timeout "$RUN_TIMEOUT" env \
        CUDA_VISIBLE_DEVICES="$gpu" TFG_SEED="$seed" EXP_TAG="$tag" \
        SKIP_CACHE=1 DFEI_VERSION=5 USE_PID=true CHANNEL="$ch" MODE=full \
        $expenv \
        bash scripts/run_nopid_bd_pipeline.sh > "$log" 2>&1
      rc=$?
      # Reap orphans: `timeout` kills the pipeline bash but not its grandchild
      # trainer.py, which would otherwise linger holding GPU memory and pile up
      # across runs (the failure mode that caused cascading timeouts). Sequential
      # queue => nothing legit is training between runs, so this is safe.
      pkill -KILL -f _auto_nopid_ift_bd 2>/dev/null || true
      pkill -KILL -f wmpgnn/analysis/trainer.py 2>/dev/null || true
      sleep 5
      if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc (see $log)"
        echo -e "$(date '+%F %T')\t$tag\trc=$rc" >> "$SUITE/failures.txt"
        continue
      fi
      if $PY scripts/suite_record.py --results "$RESULTS" \
           --exp "$exp" --channel "$ch" --seed "$seed" --log "$log"; then
        touch "$SUITE/DONE_${tag}"
        echo "[$(date '+%F %T')] OK $tag recorded"
      else
        echo "[$(date '+%F %T')] WARN $tag finished but record failed (see $log)"
        echo -e "$(date '+%F %T')\t$tag\trecord_failed" >> "$SUITE/failures.txt"
      fi
    done
  done
done

echo "############ SUITE done $(date) ############"
$PY scripts/aggregate_suite.py "$RESULTS" || true
