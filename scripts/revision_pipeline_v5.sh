#!/usr/bin/env bash
# v5 = v4 from step 2 on, with higher parallelism (GPU-light diagnostics, tuning without
# diagnostics, longest-job-first scheduling, FedQHD-GD on GPU, at most --max_heavy (default
# 16) concurrent GPU-heavy jobs). Same protocol as v4:
#   2. Acrobot re-tune (stage 1 retry pass, stage 2 K sweep) -> tuned_params (+ FedProx μ=0.01)
#   3. compile knobs + QHD-oracle tuning (missing jobs only) -> tuned_params_v3, probe spec
#   4. budget probe -> episodes_map.json
#   5. main_v3 (seeds 42-51) + anchor ablation (seeds 42-46)
#   6. audit + summaries
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ=$R/code_frozen_v4
COMMON="--bootstrap_on_truncation --anchor_m 10000 --dqn_preset zoo --tuned_qhd"
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

log "2. Acrobot stage 1 (finishing the running pass + retry)"
# jobs still running from the (stopped) v4 driver are waited for via the PID guard
drive acro_tune1_v5 44 8 --suite tune --envs Acrobot --seeds 0-2 --anchor_m 10000 \
  --bootstrap_on_truncation --no_diagnostics
python "$W/scripts/select_tuned_params.py" --stage 1
drive acro_tune2 24 6 --suite tune --tune_stage 2 --tune_best "$R/tune_best_stage1.json" --tag tune_K \
  --envs Acrobot --seeds 0-2 --anchor_m 10000 --bootstrap_on_truncation --no_diagnostics
python "$W/scripts/select_tuned_params.py" --stage 2
python - <<'EOF'
import json, os
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
for env, mu in json.load(open(f'{R}/tune_best_fedprox_mu.json')).items():   # μ = 0.01 (not tuned)
    src = f'{R}/tuned_params/{env}/q1_homogeneous/params_fedavg_dqn.json'
    p = json.load(open(src)); p['fedprox_mu'] = mu; p['method_key'] = 'fedprox_dqn'
    json.dump(p, open(src.replace('fedavg_dqn', 'fedprox_dqn'), 'w'), indent=2)
best = json.load(open(f'{R}/tune_best_stage2.json'))
missing = [f'{e}/{q}' for e in ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar'] for q in ['q1', 'q2']
           if f'{e}/{q}' not in best]
assert not missing, f'untuned cells: {missing}'
print('Acrobot tuned:', best['Acrobot/q1'], best['Acrobot/q2'])
EOF

log "3. compile knobs + QHD-oracle tuning (missing jobs only), selection"
drive tune_compile_v5 14 6 --suite tune_compile --seeds 0-2 $COMMON --no_diagnostics --max_heavy 8 \
  --params_root "$R/tuned_params" &
P1=$!
drive tune_oracle_v5 30 8 --suite tune_oracle --seeds 0-2 $COMMON --no_diagnostics --max_heavy 8 \
  --params_root "$R/tuned_params" &
P2=$!
wait $P1; wait $P2
python "$W/scripts/select_v3.py" --step tune

log "4. budget probe"
drive probe 24 8 --suite probe --probe_spec "$R/probe_spec.json" --probe_episodes 1500 \
  --seeds 0-2 $COMMON --no_diagnostics --params_root "$R/tuned_params_v3"
python "$W/scripts/select_v3.py" --step budget

log "5. main_v3 + anchor ablation"
drive driver_main_v3 46 10 --suite main --tag main_v3 --seeds 42-51 --indep_no_reset $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"
drive anchor_ablation 24 8 --suite anchor_ablation --seeds 42-46 $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"

log "6. audit + summaries"
python "$W/scripts/audit_protocol.py" --suite_dir "$R/main_v3" --tuned "$R/tuned_params_v3" \
  --episodes_map "$R/episodes_map.json" > "$R/audit_main_v3.txt"
head -1 "$R/audit_main_v3.txt"
python "$W/scripts/summarize_revision.py" --suites main_v3 anchor_ablation --main_dirs main_v3 anchor_ablation
log "pipeline v5 finished"
