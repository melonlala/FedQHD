#!/bin/bash
# ============================================================
# Hyperparameter search for LunarLander HETEROGENEOUS QHD methods
# ============================================================
# Searches: FedQHD (Hetero) and Oracle QHD (Hetero)
#
# Grid:
#   lr             = [0.1, 0.2, 0.5]
#   agg_interval   = [10, 25]
#   exploration_decay = [0.990, 0.995, 0.999]
#   rff_gamma      = 2.0  (fixed — best from homo search)
#
# Note on Oracle LR scaling:
#   Oracle QHD (Hetero) internally sets effective_lr = qhd_lr / num_agents.
#   With 2 agents: lr=0.1→0.05, lr=0.2→0.1, lr=0.5→0.25 effective.
#   This is the correct fix for the "oracle instability" bug (see MEMORY.md).
#
# Grid size : 2 methods × 3 lr × 2 agg × 3 decay = 36 combos
# Episodes  : 100   (proxy; ~⅙ of full 600)
# Agents    : 2     (faster than 5; hetero overhead is per-agent)
# Runs/combo: 1
# Est. time : ~36 × 6 min ≈ 3.5 hours  (GPU speeds this up significantly)
# ============================================================

mkdir -p logs results/hparam_search

echo "=== LunarLander Heterogeneous QHD Hyperparameter Search ==="
echo "Grid : lr=[0.1,0.2,0.5]  agg=[10,25]  decay=[0.990,0.995,0.999]  rff_gamma=2.0"
echo "Setup: 100 episodes, 2 agents, 1 run per combo"
echo "Start: $(date)"

python hparam_search.py \
    --method_type hetero \
    --env LunarLander \
    --episodes 200 \
    --agent_num 2 \
    --runs 1 \
    --lr_values "0.05, 0.1,0.2,0.5" \
    --agg_values "10,25" \
    --eps_decay_values "0.995,0.9,0.99" \
    --rff_gammas "2.0, 1.0, 0.5" \
    --exploration_rate 1.0 \
    --exploration_min 0.001 \
    --discount_factor 0.99 \
    --hyperdimension 10000 \
    --anchor_set_size 200 \
    --output_dir results/hparam_search \
    2>&1 | tee logs/hparam_LunarLander_hetero.log

echo ""
echo "Done : $(date)"
echo "Results → results/hparam_search/LunarLander_hetero_hparam_search.json"
echo ""
echo "=== Best config per method ==="
python -c "
import json
with open('results/hparam_search/LunarLander_hetero_hparam_search.json') as f:
    d = json.load(f)
results = d['results']
seen = set()
for r in results:
    if r['method'] not in seen:
        seen.add(r['method'])
        print(f\"{r['method']:25s}  lr={r['learning_rate']:<5}  agg={r['aggregation_interval']:<4}  \"
              f\"decay={r['exploration_decay']:<6}  gamma={r['rff_gamma']:<5}  reward={r['mean_reward']:.2f}\")
        if r['method'] == 'Oracle QHD (Hetero)':
            n = 2  # num_agents used in this search
            print(f\"  → effective oracle lr = {r['learning_rate']} / {n} = {r['learning_rate']/n:.4f}\")
"
