#!/bin/bash
# claude_made: GPU racers for the slowest main_v3 LunarLander DQN cells (code_frozen_v4, args unchanged except
# --dqn_device cuda and output under results/revision/gpu_race/). First finished copy of a cell wins.
# Each line of slurm/ll_dqn_gpu_cmds.txt is "<cpu-array-task-id> <command>". Touch slurm/stop/<id> to stop a racer.
#SBATCH -J ll_dqn_gpu
#SBATCH -p b200-devel
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o /home/hou_yuchen_neu/projects/FedQHD/logs/aicr/gpu_race_%j.out
source ~/.bashrc; conda activate hdc_fedrl
W=$HOME/projects/FedQHD; F=$W/results/revision/code_frozen_v4; cd $F
export PYTHONPATH=$F OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $W/slurm/stop; nvidia-smi -L
IFS=, read -ra GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1}"; echo "job GPUs: ${GPUS[*]}"
declare -A PIDS; n=0
while read -r ID CMD; do
  CMD=$(echo "$CMD" | sed -e "s#@W@#$W#g" -e "s#@DEV@#cuda#g")
  read M S V <<<$(echo "$CMD" | awk '{for(i=1;i<=NF;i++){if($i=="--method")m=$(i+1);if($i=="--random_seed")s=$(i+1);if($i=="--output_dir")o=$(i+1)}; k=split(o,a,"/"); print m,s,a[k-1]}')
  LOG=$W/logs/aicr/gpu_LunarLander_${M}_${V}_seed${S}.log
  { echo "host=$(hostname) dev=cuda gpu=${GPUS[$((n % ${#GPUS[@]}))]} start=$(date -Is)"; echo "$CMD"; } > $LOG
  ( CUDA_VISIBLE_DEVICES=${GPUS[$((n % ${#GPUS[@]}))]} python $CMD >> $LOG 2>&1; echo "exit=$? end=$(date -Is)" >> $LOG ) &
  PIDS[$ID]=$!; n=$((n+1))
done < $W/slurm/ll_dqn_gpu_cmds.txt
while [ -n "$(jobs -rp)" ]; do
  for ID in "${!PIDS[@]}"; do
    if [ -e $W/slurm/stop/$ID ] && kill -0 ${PIDS[$ID]} 2>/dev/null; then
      pkill -TERM -P ${PIDS[$ID]}; echo "stopped racer $ID $(date -Is)"; unset PIDS[$ID]
    fi
  done
  sleep 30
done
