#!/usr/bin/env bash
# Fixed-total-budget scalability with the main_v3 protocol (rule confirmed 2026-10-07):
# total client-episodes = 5 x env budget (episodes_map.json), N in {1,2,5,10}, seeds 42-46.
# Starts after revision_pipeline_v6.sh has finished (main_v3 + anchor ablation).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ=$R/code_frozen_v4
export PYTHONPATH=$W
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for pipeline v6"
until grep -qE 'pipeline v6 finished|FAILED' "$R/pipeline_v6.log" 2>/dev/null; do sleep 300; done
grep -q FAILED "$R/pipeline_v6.log" && { log "v6 failed; not starting"; exit 1; }
ARGS="--suite budget --tag budget_v3 --seeds 42-46 --n_grid 1 2 5 10 --budget_factor 5
  --episodes_map $R/episodes_map.json --dqn_preset zoo --tuned_qhd --anchor_m 10000
  --bootstrap_on_truncation --indep_no_reset --dqn_device cpu --params_root $R/tuned_params_v3"
python "$W/scripts/run_revision_experiments.py" $ARGS --jobs 46 --code_root "$FZ" > "$R/budget_v3.log" 2>&1 || true
python "$W/scripts/run_revision_experiments.py" $ARGS --jobs 10 --code_root "$FZ" > "$R/budget_v3.retry.log" 2>&1
tail -1 "$R/budget_v3.retry.log" | grep -q '^done, 0 failed' || { log "FAILED: budget_v3"; exit 1; }
python "$W/scripts/summarize_revision.py" --suites budget_v3 --main_dirs main_v3 anchor_ablation 2>&1 | tail -1 || true
log "budget_v3 finished"
