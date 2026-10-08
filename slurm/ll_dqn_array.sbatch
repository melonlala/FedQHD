#!/bin/bash
# claude_made: rerun of the slow main_v3 LunarLander DQN cells (code_frozen_v4, identical args) on aicr.
# Usage: sbatch --array=1-28 [-p PART --gres=gpu:1] slurm/ll_dqn_array.sbatch <cpu|cuda>
#SBATCH -J ll_dqn
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/hou_yuchen_neu/projects/FedQHD/logs/aicr/slurm_%A_%a.out
DEV=${1:-cpu}
source ~/.bashrc; conda activate hdc_fedrl
W=$HOME/projects/FedQHD; cd $W
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/ll_dqn_cmds.txt | sed -e "s#@W@#$W#g" -e "s#@DEV@#$DEV#g")
read M E S V <<<$(echo "$CMD" | awk '{for(i=1;i<=NF;i++){if($i=="--method")m=$(i+1);if($i=="--env")e=$(i+1);if($i=="--random_seed")s=$(i+1);if($i=="--output_dir")o=$(i+1)}; n=split(o,a,"/"); print m,e,s,a[n-1]}')
LOG=$W/logs/aicr/${E}_${M}_${V}_seed${S}.log
echo "host=$(hostname) dev=$DEV start=$(date -Is)" > $LOG
echo "$CMD" >> $LOG
F=$W/results/revision/code_frozen_v4; cd $F
export PYTHONPATH=$F OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python $CMD >> $LOG 2>&1
echo "exit=$? end=$(date -Is)" >> $LOG
