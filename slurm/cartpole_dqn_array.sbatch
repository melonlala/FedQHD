#!/bin/bash
# claude_made: main_v3 CartPole DQN-family jobs offloaded from the revision session (848afe).
# Each line of slurm/cartpole_dqn_cmds.txt runs unchanged from code_frozen_v4 with PYTHONPATH=.
# Usage: sbatch --array=1-29 slurm/cartpole_dqn_array.sbatch
#SBATCH -J cp_dqn
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o /home/hou_yuchen_neu/projects/FedQHD/logs/aicr/slurm_%A_%a.out
source ~/.bashrc; conda activate hdc_fedrl
W=$HOME/projects/FedQHD
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $W/slurm/cartpole_dqn_cmds.txt)
read M E S O <<<$(echo "$CMD" | awk '{for(i=1;i<=NF;i++){if($i=="--method")m=$(i+1);if($i=="--env")e=$(i+1);if($i=="--random_seed")s=$(i+1);if($i=="--output_dir")o=$(i+1)}; print m,e,s,o}')
V=$(echo "$O" | cut -d/ -f2)
LOG=$W/logs/aicr/${E}_${M}_${V}_seed${S}.log
cd $W/results/revision/code_frozen_v4
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) start=$(date -Is)" > $LOG
echo "$CMD" >> $LOG
$CMD >> $LOG 2>&1
echo "exit=$? end=$(date -Is)" >> $LOG
