#!/usr/bin/env bash
# Unattended revision pipeline (run from the repo root, after stage-1 tuning was launched):
#   1. wait for stage-1 FedQHD tuning      (results/revision/tune,      driver_tune1.log)
#   2. select stage-1 winners, then run in parallel
#        stage-2 K sweep                    (results/revision/tune_K)
#        FedProx-DQN μ sweep                (results/revision/tune_prox)
#   3. select K and μ -> tuned params root  (results/revision/tuned_params)
#   4. main table v2, 10 seeds              (results/revision/main_v2)
# Every step is resumable; re-running the script skips finished jobs.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ=$R/code_frozen_v2                     # code snapshot used by steps 2-4
COMMON="--bootstrap_on_truncation --anchor_m 10000"
export PYTHONPATH=$W
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -d "$FZ" ]; then
  mkdir -p "$FZ"
  rsync -a --include='*/' --include='*.py' --exclude='*' "$W/agent" "$W/env" "$W/scripts" "$W/utils" "$FZ/"
fi

log "waiting for stage-1 tuning"
until grep -q '^done,' "$R/driver_tune1.log" 2>/dev/null; do sleep 60; done
python "$W/scripts/select_tuned_params.py" --stage 1

log "stage 2 (K sweep) + FedProx mu sweep"
python "$W/scripts/run_revision_experiments.py" --suite tune --tune_stage 2 \
  --tune_best "$R/tune_best_stage1.json" --tag tune_K --seeds 0-2 $COMMON \
  --jobs 30 --code_root "$FZ" > "$R/driver_tune2.log" 2>&1 &
P1=$!
python "$W/scripts/run_revision_experiments.py" --suite tune_prox --seeds 0-2 \
  --dqn_preset zoo --bootstrap_on_truncation --jobs 14 --code_root "$FZ" \
  > "$R/driver_tune_prox.log" 2>&1 &
P2=$!
wait $P1 $P2

python "$W/scripts/select_tuned_params.py" --stage 2
python - <<'EOF'
# FedProx μ: best mean last-100 training return over tuning seeds, per environment
import glob, json, os, numpy as np
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
best = {}
for env in ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar']:
    sc = {}
    for d in glob.glob(f'{R}/tune_prox/mu*'):
        fs = glob.glob(f'{d}/seed*/{env}/q1_homogeneous/FedProx-DQN.json')
        if fs:
            sc[float(os.path.basename(d)[2:])] = np.mean(
                [np.mean(json.load(open(f))['reward_history'][-100:]) for f in fs])
    if not sc:
        continue
    mu = max(sc, key=sc.get)
    best[env] = mu
    src = f'{R}/tuned_params/{env}/q1_homogeneous/params_fedavg_dqn.json'
    p = json.load(open(src)); p['fedprox_mu'] = mu; p['method_key'] = 'fedprox_dqn'
    json.dump(p, open(src.replace('fedavg_dqn', 'fedprox_dqn'), 'w'), indent=2)
    print(env, 'mu scores', sc, '-> mu =', mu)
json.dump(best, open(f'{R}/tune_best_fedprox_mu.json', 'w'), indent=1)
EOF

log "main table v2"
python "$W/scripts/run_revision_experiments.py" --suite main --tag main_v2 --seeds 42-51 \
  --dqn_preset zoo --tuned_qhd --indep_no_reset $COMMON \
  --params_root "$R/tuned_params" --jobs 44 --code_root "$FZ" > "$R/driver_main_v2.log" 2>&1
log "pipeline finished"
