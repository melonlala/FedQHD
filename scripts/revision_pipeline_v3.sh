#!/usr/bin/env bash
# v3 protocol: one pipeline for every cell of the main table.
#   0. wait for steps 1-3 of revision_pipeline_fix_q2.sh (Q2 FedQHD tuning repaired), then stop it
#   1. tuning on seeds 0-2 at 600 episodes:
#        tune_compile  FedQHD-hetero ridge λ x {overwrite, warmstart}
#        tune_oracle   pooled-data references (QHD: lr x ε-decay; DQN: lr scale x ε-schedule scale)
#   2. select -> results/revision/tuned_params_v3, probe_spec.json
#   3. budget probe: selected oracle per (env, setting), 1500 episodes, seeds 0-2
#        -> results/revision/episodes_map.json (pre-registered 95%-of-improvement rule)
#   4. main_v3: every method, seeds 42-51, per-env budget, tuned_params_v3
#   5. protocol audit + summary
# Each driver call is followed by a low-parallelism retry pass and must end "done, 0 failed".
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ=$R/code_frozen_v3
COMMON="--bootstrap_on_truncation --anchor_m 10000 --dqn_preset zoo --tuned_qhd"
export PYTHONPATH=$W
log() { echo "[$(date '+%F %T')] $*"; }
drive() {  # drive <log-name> <jobs> <retry-jobs> <driver args...>
  local name=$1 jobs=$2 retry=$3; shift 3
  python "$W/scripts/run_revision_experiments.py" "$@" --jobs "$jobs" --code_root "$FZ" \
    > "$R/$name.log" 2>&1 || true
  python "$W/scripts/run_revision_experiments.py" "$@" --jobs "$retry" --code_root "$FZ" \
    > "$R/$name.retry.log" 2>&1
  # the driver always ends with "done, <n> failed" (also when nothing was pending)
  tail -1 "$R/$name.retry.log" | grep -q '^done, 0 failed' \
    || { log "FAILED: $name"; tail -3 "$R/$name.retry.log"; exit 1; }
  log "$name complete"
}

log "0. waiting for the Q2 tuning repair (steps 1-3)"
until grep -qE '4\. wait for main_v2|FAILED' "$R/pipeline_fix.log"; do sleep 120; done
grep -q FAILED "$R/pipeline_fix.log" && { log "repair failed"; exit 1; }
P=$(pgrep -f "[r]evision_pipeline_fix_q2.sh" || true); [ -n "$P" ] && kill $P || true
log "repair steps 1-3 done; stopped its main_v2 step (superseded by main_v3)"

mkdir -p "$FZ"
rsync -a --include='*/' --include='*.py' --exclude='*' "$W/agent" "$W/env" "$W/scripts" "$W/utils" "$FZ/"

log "1. tuning (compile knobs + pooled oracles)"
drive tune_compile 10 4 --suite tune_compile --seeds 0-2 $COMMON --params_root "$R/tuned_params" &
P1=$!
drive tune_oracle 30 6 --suite tune_oracle --seeds 0-2 $COMMON --params_root "$R/tuned_params" &
P2=$!
wait $P1; wait $P2

log "2. selection"
python "$W/scripts/select_v3.py" --step tune

log "3. budget probe"
drive probe 24 6 --suite probe --probe_spec "$R/probe_spec.json" --probe_episodes 1500 \
  --seeds 0-2 $COMMON --params_root "$R/tuned_params_v3"
python "$W/scripts/select_v3.py" --step budget

log "4. main_v3"
drive driver_main_v3 40 8 --suite main --tag main_v3 --seeds 42-51 --indep_no_reset $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"

log "5. audit + summary"
python "$W/scripts/audit_protocol.py" --suite_dir "$R/main_v3" --tuned "$R/tuned_params_v3" \
  --episodes_map "$R/episodes_map.json" > "$R/audit_main_v3.txt"
head -1 "$R/audit_main_v3.txt"
python "$W/scripts/summarize_revision.py" --suites main_v3 --main_dirs main_v3
log "pipeline v3 finished"
