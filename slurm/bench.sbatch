#!/bin/bash
#SBATCH -J fedqhd_bench
#SBATCH -p rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:25:00
#SBATCH -o /home/hou_yuchen_neu/projects/FedQHD/logs/aicr/bench_%j.out
source ~/.bashrc; conda activate hdc_fedrl
W=~/projects/FedQHD; F=$W/results/revision/code_frozen_v4; cd $F; export PYTHONPATH=$F; nvidia-smi -L
ARGS="--method distillation_dqn --env LunarLander --question q1 --random_seed 42 --runs 1 --episodes 1500 --agent_num 5 --qhd_lr 0.1 --qhd_discount 0.95 --qhd_exploration_rate 1.0 --qhd_exploration_decay 0.95 --qhd_exploration_min 0.001 --qhd_agg_interval 25 --dqn_lr 0.001 --dqn_discount 0.99 --dqn_exploration_rate 1.0 --dqn_exploration_decay 0.995 --dqn_exploration_min 0.001 --dqn_agg_interval 25 --hyperdimension 10000 --rff_gamma 1.0 --anchor_set_size 200 --bootstrap_on_truncation --indep_no_reset --dqn_preset zoo"
for dev in cuda cpu; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 900 python $W/results/revision/code_frozen_v4/scripts/run_single_method.py $ARGS --output_dir /tmp/bench_$dev --dqn_device $dev > $W/logs/aicr/bench_${SLURM_JOB_ID}_$dev.log 2>&1 &
done
wait
