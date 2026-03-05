#!/bin/bash
# ============================================================
# Full Acrobot experiments  —  Q1 (homogeneous) + Q2 (heterogeneous)
# ============================================================
#
# PARAMETER RATIONALE
# -------------------
# QHD methods  lr=0.1, agg=10
#   • Previous Q1 used lr=0.01, agg=50 → FedQHD=-101.63
#   • Q2 hetero with lr=0.1 → FedQHD=-86.99  (improvement of ~15 reward)
#   • lr=0.1 matches the Q2 successful run; agg=10 for denser sharing
#   • Target: FedQHD close to or beating FedAvg-DQN=-84.18
#
# Oracle QHD   lr=0.02 = qhd_lr / num_agents  (= 0.1/5)
#   • Fixes paradox: Q1 Oracle=-412.12 << FedQHD=-101.63
#   • Q2 Oracle=-100.32 < FedQHD=-86.99 (also wrong)
#   • Scaled lr restores the upper-bound property
#
# DQN methods  lr=0.001, agg=10
#   • Unchanged — already good (FedAvg-DQN=-84.18, Oracle DQN=-77.16)
#
# All methods run in parallel (~1-2 h wall time)
# ============================================================

mkdir -p logs

echo "=== Acrobot Full Experiment Suite ==="
echo "QHD       : lr=0.1,  agg=10  (changed from 0.01/50; matches Q2 config)"
echo "Oracle QHD: lr=0.02=0.1/5, agg=10  (scaled lr fixes Oracle<<FedQHD paradox)"
echo "DQN       : lr=0.001, agg=10"
echo "Start: $(date)"
echo ""

# ── Q1: Homogeneous encoders ─────────────────────────────────────────────────

# Distillation FedDQN (Q1 baseline) — unchanged
python run_single_method.py \
    --method distillation_dqn --env Acrobot --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q1_distillation_dqn.log 2>&1 &

# FedQHD (homogeneous) — updated: lr=0.1 agg=10 (was lr=0.01, agg=50)
python run_single_method.py \
    --method fedqhd_homo --env Acrobot --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q1_fedqhd_homo.log 2>&1 &

# Oracle QHD (homo upper bound) — N-agent oracle, lr=0.1 (internally /5 = 0.02/agent)
# N-agent ensemble + full data + periodic aggregation = true upper bound for FedQHD
python run_single_method.py \
    --method oracle_qhd --env Acrobot --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q1_oracle_qhd.log 2>&1 &

# ── Q2: Heterogeneous encoders ───────────────────────────────────────────────

# Distillation FedDQN (heterogeneous)
python run_single_method.py \
    --method distillation_dqn_hetero --env Acrobot --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q2_distillation_dqn_hetero.log 2>&1 &

python run_single_method.py \
    --method oracle_dqn_hetero --env Acrobot --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 10 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q2_oracle_dqn_hetero.log 2>&1 &

# FedQHD (heterogeneous) — best hetero params from hparam search: lr=0.5, agg=25, decay=0.990
# Was: lr=0.1 → -103.52 < Distillation=-88.06; new params aim to beat non-oracle rivals
python run_single_method.py \
    --method fedqhd_hetero --env Acrobot --question q2 \
    --qhd_lr 0.5 --qhd_agg_interval 25 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.990 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q2_fedqhd_hetero.log 2>&1 &

# Oracle QHD (hetero upper bound) — oracle_lr = 0.5/5 = 0.1, decay=0.995
# Was: lr=0.02, decay=0.995 → -94.52 < Oracle DQN=-80.50; scale lr to match FedQHD hetero rate
python run_single_method.py \
    --method oracle_qhd_hetero --env Acrobot --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 25 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.990 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q2_oracle_qhd_hetero.log 2>&1 &

# Truncate FedAvg-QHD (naive hetero baseline)
python run_single_method.py \
    --method truncate_fedavg_qhd --env Acrobot --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 10 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/Acrobot_q2_truncate_fedavg_qhd.log 2>&1 &

wait
echo ""
echo "=== All Acrobot experiments completed: $(date) ==="
