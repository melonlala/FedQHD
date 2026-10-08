#!/usr/bin/env bash
# v6 = steps 5-6 of v5 (tuning, selection and budget probe already finished), with the
# DQN family on CPU (--dqn_device cpu): ~4x faster per gradient step than the shared GPU.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ=$R/code_frozen_v4
COMMON="--bootstrap_on_truncation --anchor_m 10000 --dqn_preset zoo --tuned_qhd --dqn_device cpu"
export PYTHONPATH=$W
log() { echo "[$(date '+%F %T')] $*"; }
drive() {  # drive <log-name> <jobs> <retry-jobs> <driver args...>
  local name=$1 jobs=$2 retry=$3; shift 3
  python "$W/scripts/run_revision_experiments.py" "$@" --jobs "$jobs" --code_root "$FZ" \
    > "$R/$name.log" 2>&1 || true
  python "$W/scripts/run_revision_experiments.py" "$@" --jobs "$retry" --code_root "$FZ" \
    > "$R/$name.retry.log" 2>&1
  tail -1 "$R/$name.retry.log" | grep -q '^done, 0 failed' \
    || { log "FAILED: $name"; tail -3 "$R/$name.retry.log"; exit 1; }
  log "$name complete"
}
log "5. main_v3 + anchor ablation (episodes: $(cat $R/episodes_map.json | tr -d '\n '))"
drive driver_main_v3 46 10 --suite main --tag main_v3 --seeds 42-46 --indep_no_reset $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"
drive anchor_ablation 24 8 --suite anchor_ablation --seeds 42-46 $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"
log "6. audit + summaries"
python "$W/scripts/audit_protocol.py" --suite_dir "$R/main_v3" --tuned "$R/tuned_params_v3" \
  --episodes_map "$R/episodes_map.json" --seeds 42-46 > "$R/audit_main_v3.txt"
head -1 "$R/audit_main_v3.txt"
python "$W/scripts/summarize_revision.py" --suites main_v3 anchor_ablation --main_dirs main_v3 anchor_ablation
log "pipeline v6 finished"
