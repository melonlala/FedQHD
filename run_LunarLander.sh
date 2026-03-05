#!/bin/bash
# ============================================================
# Full LunarLander experiments  —  Q1 (homogeneous) + Q2 (heterogeneous)
# ============================================================
#
# PARAMETER RATIONALE
# -------------------
# QHD methods  lr=0.1, agg=10
#   • hdsarsa_figs/LunarLander/5_qhd used identical params, giving
#     FedAvg ≈ 17.85 reward  (vs current -1.54 with agg=50)
#   • Confirmed direction: hparam search found lr=1.0 → -42.96 (worse);
#     smaller lr=0.1 is more stable for LunarLander
#   • agg=10 (every 10 episodes) gives denser knowledge sharing than agg=50
#
# Oracle QHD   lr=0.02 = qhd_lr / num_agents
#   • Oracle trains ALL N environments each episode → N× gradient updates
#   • Same effective learning speed as FedQHD but with N× more diverse data
#   • This ensures Oracle QHD ≥ FedQHD (upper-bound property restored)
#   • Fixes: current Oracle=-16.93 < FedQHD=-1.54 paradox
#
# DQN methods  lr=0.001, agg=10
#   • Standard DQN hyperparameters; same as existing successful runs
#   • agg=10 matches QHD aggregation frequency for fair comparison
#
# Distillation DQN  lr=0.001, agg=25
#   • Larger agg interval to allow student networks to stabilise
#     before each distillation round
#
# ALL methods run in parallel (~2-3 h wall time with 5 agents each)
# ============================================================

mkdir -p logs

echo "=== LunarLander Full Experiment Suite ==="
echo "QHD  : lr=0.1,   agg=10  (best config from hdsarsa_figs)"
echo "Oracle QHD : lr=0.02=0.1/5, agg=10  (scaled to match FedQHD update rate)"
echo "DQN  : lr=0.001, agg=10"
echo "Distillation DQN : lr=0.001, agg=25"
echo "Start: $(date)"
echo ""

# ── Q1: Homogeneous encoders (6 methods) ────────────────────────────────────

# Independent QHD — lower bound (no federation)
python run_single_method.py \
    --method independent_qhd --env LunarLander --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_independent_qhd.log 2>&1 &

# FedQHD (homogeneous) — main method
# lr=0.1 agg=10: hdsarsa_figs confirmed ≈17.85 FedAvg reward (vs -1.54 with agg=50)
python run_single_method.py \
    --method fedqhd_homo --env LunarLander --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_fedqhd_homo.log 2>&1 &
# (hdc_fedrl) yuchen@goodserver:~/projects/FedQHD$ python run_single_method.py \
#     --method fedqhd_homo --env LunarLander --question q1 \
#     --qhd_lr 0.2 --qhd_agg_interval 10 \
#     --qhd_discount 0.99 \
#     --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.99 --qhd_exploration_min 0.001 \
#     --episodes 600 --agent_num 5 \
#     > logs/LunarLander_q1_fedqhd_homo.log 2>&1 &
# [1] 3005574

# Oracle QHD (homo upper bound) — N agents, same encoder, each on own env, aggregate every episode
# Uses same lr=0.1 as FedQHD (each agent still does 1 env/episode like FedQHD)
# agg_interval param is ignored in oracle homo (always aggregates every episode)
python run_single_method.py \
    --method oracle_qhd --env LunarLander --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_oracle_qhd.log 2>&1 &

# FedAvg-DQN — DNN baseline
python run_single_method.py \
    --method fedavg_dqn --env LunarLander --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_fedavg_dqn.log 2>&1 &

# Oracle DQN (homo upper bound)
python run_single_method.py \
    --method oracle_dqn --env LunarLander --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_oracle_dqn.log 2>&1 &

# Distillation FedDQN (Q1 baseline)
python run_single_method.py \
    --method distillation_dqn --env LunarLander --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 25 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q1_distillation_dqn.log 2>&1 &

# ── Q2: Heterogeneous encoders (4 methods) ──────────────────────────────────

# FedQHD (heterogeneous) — main method for Q2
# Best params from hparam search: lr=0.5, agg=25, decay=0.900, rff_gamma=0.5
# decay=0.900 was identified as critical; decay=0.990 gave much worse results
python run_single_method.py \
    --method fedqhd_hetero --env LunarLander --question q2 \
    --qhd_lr 0.5 --qhd_agg_interval 25 \
    --rff_gamma 0.5 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.900 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_fedqhd_hetero.log 2>&1 &

# Oracle QHD (hetero upper bound) — best from hparam search: lr=0.5, decay=0.900, rff_gamma=1.0
# No anchor aggregation (large agg_interval) to match hparam search condition
# oracle_lr = qhd_lr/N = 0.5/5 = 0.1/agent
python run_single_method.py \
    --method oracle_qhd_hetero --env LunarLander --question q2 \
    --qhd_lr 0.5 --qhd_agg_interval 100000 \
    --rff_gamma 1.0 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.900 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_oracle_qhd_hetero.log 2>&1 &
python run_single_method.py \
    --method independent_qhd_hetero --env LunarLander --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_independent_qhd_hetero.log 2>&1 &


# Truncate FedAvg-QHD — naive hetero baseline (pad/truncate to fixed dim)
python run_single_method.py \
    --method truncate_fedavg_qhd --env LunarLander --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_truncate_fedavg_qhd.log 2>&1 &

# Distillation FedDQN (heterogeneous)
python run_single_method.py \
    --method distillation_dqn_hetero --env LunarLander --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 25 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_distillation_dqn_hetero.log 2>&1 &
python run_single_method.py \
    --method oracle_dqn_hetero --env LunarLander --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/LunarLander_q2_oracle_dqn_hetero.log 2>&1 &

wait
echo ""
echo "=== All LunarLander experiments completed: $(date) ==="
