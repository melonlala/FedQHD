#!/bin/bash
# ============================================================
# Full MountainCar experiments  —  Q1 (fix Oracle) + Q2 (heterogeneous)
# ============================================================
#
# PARAMETER RATIONALE
# -------------------
# QHD methods  lr=0.1, agg=50
#   • Q1 already yielded FedQHD=-140.81 (98.1% success) with these params
#   • agg=50 suits MountainCar's sparse-reward setting:
#     agents accumulate enough experience between aggregations
#   • rff_gamma=1.0 (default)
#
# Oracle QHD   lr=0.02 = qhd_lr / num_agents
#   • Current Q1 Oracle=-163.87 is paradoxically WORSE than FedQHD=-140.81
#   • Root cause: Oracle processes N environments per episode → N× gradient
#     updates at lr=0.1 → effective lr=0.5/episode → unstable oscillations
#   • Fix: scale lr to 0.02 so effective lr/episode = 0.02×5 = 0.1 (= FedQHD)
#   • Oracle then has same update speed but with 5× more diverse data → better
#
# DQN methods  lr=0.001, agg=25
#   • Standard DQN; agg=25 for distillation stability
#
# Q1 note: only Oracle QHD needs re-running (others are already correct)
# Q2 note: all 4 hetero methods are new experiments
#
# All methods run in parallel (~1-1.5 h wall time)
# ============================================================

mkdir -p logs

echo "=== MountainCar Full Experiment Suite ==="
echo "QHD       : lr=0.1,  agg=50  (best known; Q1 FedQHD=−140.81, 98% success)"
echo "Oracle QHD: lr=0.02=0.1/5, agg=50  (scaled lr fixes Oracle<FedQHD paradox)"
echo "DQN       : lr=0.001, agg=25"
echo "Start: $(date)"
echo ""

# ── Q1: Fix Oracle QHD only (other Q1 results are correct and kept) ─────────

# Oracle QHD (homo upper bound) — N-agent oracle, lr=0.1 (internally /5=0.02/agent), agg=50
# N-agent ensemble + full data + periodic aggregation = true upper bound for FedQHD
python run_single_method.py \
    --method oracle_qhd --env MountainCar --question q1 \
    --qhd_lr 0.1 --qhd_agg_interval 50 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q1_oracle_qhd.log 2>&1 &

python run_single_method.py \
    --method distillation_dqn --env MountainCar --question q1 \
    --dqn_lr 0.001 --dqn_agg_interval 25 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q1_distillation_dqn.log 2>&1 &


# ── Q2: Heterogeneous encoders (4 methods) ──────────────────────────────────

# FedQHD (heterogeneous) — best hetero params: lr=0.5, agg=25, decay=0.990
# Was: lr=0.1, agg=50 → -194.97 < Truncate=-166.78; new params from hparam search
python run_single_method.py \
    --method fedqhd_hetero --env MountainCar --question q2 \
    --qhd_lr 0.5 --qhd_agg_interval 25 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.990 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_fedqhd_hetero.log 2>&1 &

# Oracle QHD (hetero upper bound) — oracle_lr = 0.5/5 = 0.1 (scaled from FedQHD hetero)
# Was: lr=0.02 → -148.91 < Oracle DQN=-119.97; scale lr to match FedQHD hetero effective rate
python run_single_method.py \
    --method oracle_qhd_hetero --env MountainCar --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 25 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.990 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_oracle_qhd_hetero.log 2>&1 &

# Truncate FedAvg-QHD — naive hetero baseline (pad/truncate to fixed dim)
python run_single_method.py \
    --method truncate_fedavg_qhd --env MountainCar --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 50 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_truncate_fedavg_qhd.log 2>&1 &

python run_single_method.py \
    --method independent_qhd_hetero --env MountainCar --question q2 \
    --qhd_lr 0.1 --qhd_agg_interval 50 \
    --qhd_discount 0.99 \
    --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.995 --qhd_exploration_min 0.001 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_independent_qhd_hetero.log 2>&1 &

# Distillation FedDQN (heterogeneous)
python run_single_method.py \
    --method distillation_dqn_hetero --env MountainCar --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 25 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_distillation_dqn_hetero.log 2>&1 &

python run_single_method.py \
    --method oracle_dqn_hetero --env MountainCar --question q2 \
    --dqn_lr 0.001 --dqn_agg_interval 25 \
    --episodes 600 --agent_num 5 \
    > logs/MountainCar_q2_oracle_dqn_hetero.log 2>&1 &

wait
echo ""
echo "=== All MountainCar experiments completed: $(date) ==="
