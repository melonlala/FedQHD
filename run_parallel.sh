#!/usr/bin/env bash
# Run FedQHD experiments in parallel with nohup
# Logs go to logs/<env>_<experiment>.log, PIDs saved to logs/pids.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/pids.txt"

# ── Configuration ──────────────────────────────────────────────────────────────
AGENT_NUM=5
RUNS=3
HYPERDIMENSION=10000
OUTPUT_DIR="results"
PYTHON="${PYTHON:-python3}"

ENVS=(CartPole Acrobot LunarLander MountainCar)
EXPERIMENTS=(q1 q2)

# Per-environment hyperparameters from hdsarsa_figs experiment_info.txt
# Columns: episodes  qhd_lr  qhd_agg  qhd_discount  qhd_eps  qhd_eps_decay  dqn_lr  dqn_agg
declare -A ENV_PARAMS
ENV_PARAMS[CartPole]="1000  0.1  25   0.9   0.8  0.9    0.001  10"
ENV_PARAMS[Acrobot]="600   0.2  5   0.99  1.0  0.995  0.001  50"
ENV_PARAMS[LunarLander]="600   0.1  10  0.9   1.0  0.995  0.001  5"
ENV_PARAMS[MountainCar]="600   0.2  30  0.99  1.0  0.995  0.001  50"
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [COMMAND]

Commands:
  start   Launch all experiments in parallel (default)
  status  Show status of running/completed experiments
  kill    Kill all tracked experiment processes
  logs    Tail all log files
EOF
    exit 0
}

start_experiments() {
    mkdir -p "$LOG_DIR"
    > "$PID_FILE"

    echo "=================================================="
    echo " FedQHD Parallel Experiment Launcher"
    echo "=================================================="
    echo " Environments : ${ENVS[*]}"
    echo " Experiments  : ${EXPERIMENTS[*]}"
    echo " Agents       : $AGENT_NUM  |  Runs: $RUNS"
    echo " Hyperparams  : per-environment (from hdsarsa_figs)"
    echo "=================================================="
    printf "  %-14s %6s  %6s %5s  %5s %6s %10s  %7s %7s\n" \
        "ENV" "eps" "qhd_lr" "qAgg" "qDisc" "qEps" "qEpsDcy" "dqn_lr" "dAgg"
    for ENV in "${ENVS[@]}"; do
        read -r EP QLR QAGG QDISC QEPS QEPSD DLR DAGG <<< "${ENV_PARAMS[$ENV]}"
        printf "  %-14s %6s  %6s %5s  %5s %6s %10s  %7s %7s\n" \
            "$ENV" "$EP" "$QLR" "$QAGG" "$QDISC" "$QEPS" "$QEPSD" "$DLR" "$DAGG"
    done
    echo ""

    for ENV in "${ENVS[@]}"; do
        read -r EP QLR QAGG QDISC QEPS QEPSD DLR DAGG <<< "${ENV_PARAMS[$ENV]}"
        for EXP in "${EXPERIMENTS[@]}"; do
            LOG="$LOG_DIR/${ENV}_${EXP}.log"
            CMD="$PYTHON $SCRIPT_DIR/run_experiments.py \
                --experiment $EXP \
                --env $ENV \
                --episodes $EP \
                --agent_num $AGENT_NUM \
                --runs $RUNS \
                --hyperdimension $HYPERDIMENSION \
                --anchor_set_size 200 \
                --output_dir $OUTPUT_DIR \
                --qhd_lr $QLR \
                --qhd_agg_interval $QAGG \
                --qhd_discount $QDISC \
                --qhd_exploration_rate $QEPS \
                --qhd_exploration_decay $QEPSD \
                --dqn_lr $DLR \
                --dqn_agg_interval $DAGG"
            nohup $CMD > "$LOG" 2>&1 &
            PID=$!
            echo "$PID $ENV $EXP $LOG" >> "$PID_FILE"
            echo "  [PID $PID] $ENV / $EXP  →  $LOG"
        done
    done

    echo ""
    echo "All experiments launched. Monitor with:"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  $0 kill"
}

status_experiments() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found. Have you run '$0 start'?"
        exit 1
    fi

    printf "%-8s %-14s %-12s %-10s %s\n" "PID" "ENV" "EXPERIMENT" "STATUS" "LOG"
    printf "%-8s %-14s %-12s %-10s %s\n" "---" "---" "----------" "------" "---"
    while read -r PID ENV EXP LOG; do
        if kill -0 "$PID" 2>/dev/null; then
            STATUS="RUNNING"
        else
            if grep -q "EXPERIMENTS COMPLETE" "$LOG" 2>/dev/null; then
                STATUS="DONE"
            else
                STATUS="FAILED/EXIT"
            fi
        fi
        printf "%-8s %-14s %-12s %-10s %s\n" "$PID" "$ENV" "$EXP" "$STATUS" "$LOG"
    done < "$PID_FILE"
}

kill_experiments() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found."
        exit 1
    fi
    echo "Killing all experiment processes..."
    while read -r PID ENV EXP _LOG; do
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "  Killed PID $PID ($ENV / $EXP)"
        else
            echo "  PID $PID ($ENV / $EXP) already stopped"
        fi
    done < "$PID_FILE"
}

tail_logs() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found."
        exit 1
    fi
    LOG_FILES=()
    while read -r _PID _ENV _EXP LOG; do
        LOG_FILES+=("$LOG")
    done < "$PID_FILE"
    tail -f "${LOG_FILES[@]}"
}

COMMAND="${1:-start}"
case "$COMMAND" in
    start)  start_experiments ;;
    status) status_experiments ;;
    kill)   kill_experiments ;;
    logs)   tail_logs ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $COMMAND"; usage ;;
esac
