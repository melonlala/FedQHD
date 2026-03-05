#!/bin/bash

mkdir -p logs

python run_single_method.py --method truncate_fedavg_qhd --env CartPole --question q1 --qhd_lr 0.2 --qhd_agg_interval 50 \
    > logs/CartPole_q1_truncate_fedavg_qhd.log 2>&1 &
python run_single_method.py --method distillation_dqn --env CartPole --question q1 --dqn_lr 0.001 --dqn_agg_interval 25 \
    > logs/CartPole_q1_distillation_dqn.log 2>&1 &

python run_single_method.py --method oracle_dqn_hetero --env CartPole --question q2 --dqn_lr 0.001 --dqn_agg_interval 25 \
    > logs/CartPole_q2_oracle_dqn_hetero.log 2>&1 &

python run_single_method.py --method indepenent_qhd_hetero

wait
echo "All CartPole experiments completed."
