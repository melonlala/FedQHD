#!/usr/bin/env bash
# Repair for revision_pipeline.sh: 354 stage-1 tuning jobs of heterogeneous FedQHD failed with
# CUDA OOM (too many concurrent m=10000 jobs), so Q2 of CartPole/Acrobot/MountainCar fell back to
# the untuned params files. This script
#   1. re-runs the failed stage-1 jobs with limited parallelism,
#   2. re-selects stage 1 and runs the missing stage-2 K sweeps,
#   3. rebuilds results/revision/tuned_params (re-adding the tuned FedProx μ),
#   4. after main_v2 finishes, deletes and re-runs the main_v2 Q2 QHD-family results of the
#      affected environments with the tuned parameters.
# Every driver step must finish with "done, 0 failed", otherwise the script stops.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
R=$W/results/revision
FZ_TUNE=$R/code_frozen_tune
FZ=$R/code_frozen_v2
AFFECTED="CartPole Acrobot MountainCar"
export PYTHONPATH=$W
log() { echo "[$(date '+%F %T')] $*"; }
check() { tail -1 "$1" | grep -q '^done, 0 failed' || { log "FAILED: $1"; tail -3 "$1"; exit 1; }; }

[ -f "$R/tune_best_stage2_before_fix.json" ] || cp "$R/tune_best_stage2.json" "$R/tune_best_stage2_before_fix.json"

log "1. re-run failed stage-1 jobs (heterogeneous, 10 parallel)"
cp "$W/scripts/run_revision_experiments.py" "$FZ_TUNE/scripts/"
python "$W/scripts/run_revision_experiments.py" --suite tune --seeds 0-2 --anchor_m 10000 \
  --bootstrap_on_truncation --methods fedqhd_hetero --jobs 10 --code_root "$FZ_TUNE" \
  > "$R/driver_tune1_fix.log" 2>&1
check "$R/driver_tune1_fix.log"
python "$W/scripts/select_tuned_params.py" --stage 1

log "2. stage-2 K sweep for the missing cells"
python "$W/scripts/run_revision_experiments.py" --suite tune --tune_stage 2 \
  --tune_best "$R/tune_best_stage1.json" --tag tune_K --seeds 0-2 --anchor_m 10000 \
  --bootstrap_on_truncation --jobs 10 --code_root "$FZ" > "$R/driver_tune2_fix.log" 2>&1
check "$R/driver_tune2_fix.log"

log "3. rebuild tuned params"
python "$W/scripts/select_tuned_params.py" --stage 2
python - <<'EOF'
import json, os
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
for env, mu in json.load(open(f'{R}/tune_best_fedprox_mu.json')).items():
    src = f'{R}/tuned_params/{env}/q1_homogeneous/params_fedavg_dqn.json'
    p = json.load(open(src)); p['fedprox_mu'] = mu; p['method_key'] = 'fedprox_dqn'
    json.dump(p, open(src.replace('fedavg_dqn', 'fedprox_dqn'), 'w'), indent=2)
    print('fedprox', env, mu)
missing = [k for k in [f'{e}/{q}' for e in ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar']
                       for q in ['q1', 'q2']] if k not in json.load(open(f'{R}/tune_best_stage2.json'))]
assert not missing, f'untuned cells: {missing}'
EOF
# LunarLander/q2 was tuned before the fix; re-run it only if its selected config changed
if python - <<'EOF'
import json, os, sys
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision')
old = json.load(open(f'{R}/tune_best_stage2_before_fix.json')).get('LunarLander/q2')
new = json.load(open(f'{R}/tune_best_stage2.json')).get('LunarLander/q2')
print('LunarLander/q2 before:', old, '\nafter: ', new)
sys.exit(0 if old != new else 1)
EOF
then AFFECTED="$AFFECTED LunarLander"; fi
log "environments to re-run: $AFFECTED"

log "4. wait for main_v2, then re-run Q2 QHD-family results of: $AFFECTED"
until grep -q '^done,' "$R/driver_main_v2.log" 2>/dev/null; do sleep 120; done
QHD_Q2="independent_qhd_hetero fedqhd_hetero oracle_qhd_hetero truncate_fedavg_qhd_hetero fedqhd_gd_hetero fedhql_qhd_hetero"
python - "$AFFECTED" <<'EOF'
import glob, os, sys
R = os.path.join(os.environ['PYTHONPATH'], 'results/revision/main_v2')
names = ['Independent_QHD_(Heterogeneous)', 'FedQHD_(Heterogeneous)', 'Oracle_QHD_(Heterogeneous)',
         'Truncate_FedAvg-QHD_(Heterogeneous)', 'FedQHD-GD_(Heterogeneous)', 'FedHQL_(QHD_clients)']
n = 0
for env in sys.argv[1].split():
    for name in names:
        for f in glob.glob(f'{R}/default/seed*/{env}/q2_heterogeneous/{name}.json'):
            os.rename(f, f + '.stale_params'); n += 1
print('moved aside', n, 'stale results')
EOF
python "$W/scripts/run_revision_experiments.py" --suite main --tag main_v2 --seeds 42-51 \
  --dqn_preset zoo --tuned_qhd --indep_no_reset --bootstrap_on_truncation --anchor_m 10000 \
  --params_root "$R/tuned_params" --envs $AFFECTED --methods $QHD_Q2 --jobs 12 \
  --code_root "$FZ" > "$R/driver_main_v2_fix.log" 2>&1
check "$R/driver_main_v2_fix.log"
log "repair finished"
