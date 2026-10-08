#!/usr/bin/env bash
# v4 = v3 protocol + Acrobot bounds fix (2026-10-07) + anchor ablation.
#   0. wait for the tune_compile / tune_oracle drivers started by v3 (no duplicate jobs)
#   1. snapshot the fixed code; set aside every Acrobot QHD tuning result (wrong bounds)
#   2. re-tune Acrobot: stage 1 (agent grid) + stage 2 (K), rebuild tuned_params (+ FedProx μ)
#   3. finish tune_compile / tune_oracle (only missing jobs run), select -> tuned_params_v3
#   4. budget probe -> episodes_map.json
#   5. main_v3 (seeds 42-51) and the anchor ablation (rollout / uniform / mix, seeds 42-46)
#   6. protocol audit + summaries
# Each driver call has a low-parallelism retry pass and must end with "done, 0 failed".
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

log "0. waiting for running tuning drivers"
while pgrep -f "[r]un_revision_experiments.py --suite tune" >/dev/null; do sleep 120; done

log "1. code snapshot (fixed Acrobot bounds) and set-aside of Acrobot QHD tuning results"
mkdir -p "$FZ"
rsync -a --include='*/' --include='*.py' --exclude='*' "$W/agent" "$W/env" "$W/scripts" "$W/utils" "$FZ/"
grep -q "fixed 2026-10-07" "$FZ/env/bounds.py" || { log "snapshot lacks the bounds fix"; exit 1; }
python - <<'EOF'
import glob, os
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
pats = [f'{R}/tune/*/seed*/Acrobot', f'{R}/tune_K/*/seed*/Acrobot', f'{R}/tune_compile/*/seed*/Acrobot',
        f'{R}/tune_oracle/pooled_qhd_*/seed*/Acrobot', f'{R}/tune_oracle/oracle_qhd_*/seed*/Acrobot']
n = 0
for pat in pats:
    for d in glob.glob(pat):
        os.rename(d, d + '.bugbounds'); n += 1
print('set aside', n, 'Acrobot QHD result directories (wrong bounds)')
EOF

log "2. re-tune Acrobot (stage 1 + stage 2)"
drive acro_tune1 30 6 --suite tune --envs Acrobot --seeds 0-2 --anchor_m 10000 --bootstrap_on_truncation
python "$W/scripts/select_tuned_params.py" --stage 1
drive acro_tune2 12 4 --suite tune --tune_stage 2 --tune_best "$R/tune_best_stage1.json" --tag tune_K \
  --envs Acrobot --seeds 0-2 --anchor_m 10000 --bootstrap_on_truncation
python "$W/scripts/select_tuned_params.py" --stage 2
python - <<'EOF'
import json, os
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
for env, mu in json.load(open(f'{R}/tune_best_fedprox_mu.json')).items():
    src = f'{R}/tuned_params/{env}/q1_homogeneous/params_fedavg_dqn.json'
    p = json.load(open(src)); p['fedprox_mu'] = mu; p['method_key'] = 'fedprox_dqn'
    json.dump(p, open(src.replace('fedavg_dqn', 'fedprox_dqn'), 'w'), indent=2)
best = json.load(open(f'{R}/tune_best_stage2.json'))
missing = [f'{e}/{q}' for e in ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar'] for q in ['q1', 'q2']
           if f'{e}/{q}' not in best]
assert not missing, f'untuned cells: {missing}'
print('Acrobot tuned:', best['Acrobot/q1'], best['Acrobot/q2'])
EOF

log "3. compile knobs + pooled oracles (missing jobs only), selection"
drive tune_compile_v4 10 4 --suite tune_compile --seeds 0-2 $COMMON --params_root "$R/tuned_params" &
P1=$!
drive tune_oracle_v4 30 6 --suite tune_oracle --seeds 0-2 $COMMON --params_root "$R/tuned_params" &
P2=$!
wait $P1; wait $P2
python "$W/scripts/select_v3.py" --step tune

log "4. budget probe"
drive probe 24 6 --suite probe --probe_spec "$R/probe_spec.json" --probe_episodes 1500 \
  --seeds 0-2 $COMMON --params_root "$R/tuned_params_v3"
python "$W/scripts/select_v3.py" --step budget

log "5. main_v3 + anchor ablation"
drive driver_main_v3 40 8 --suite main --tag main_v3 --seeds 42-51 --indep_no_reset $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"
drive anchor_ablation 16 6 --suite anchor_ablation --seeds 42-46 $COMMON \
  --params_root "$R/tuned_params_v3" --episodes_map "$R/episodes_map.json"

log "6. audit + summaries"
python "$W/scripts/audit_protocol.py" --suite_dir "$R/main_v3" --tuned "$R/tuned_params_v3" \
  --episodes_map "$R/episodes_map.json" > "$R/audit_main_v3.txt"
head -1 "$R/audit_main_v3.txt"
python "$W/scripts/summarize_revision.py" --suites main_v3 anchor_ablation --main_dirs main_v3 anchor_ablation
log "pipeline v4 finished"
