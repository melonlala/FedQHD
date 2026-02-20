#!/usr/bin/env bash
# Hyperparameter search launcher — runs QHD and DQN searches in parallel
# Logs: logs/hparam_<ENV>_<TYPE>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/hparam_pids.txt"

# ── Configuration ──────────────────────────────────────────────────────────────
EPISODES=500        # shorter for faster search
AGENT_NUM=3         # fewer agents for speed
RUNS=2              # runs per combo to average
PYTHON="${PYTHON:-python3}"

# Environments and method types to search
ENVS=(CartPole Acrobot)
TYPES=(qhd dqn)
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [COMMAND]

Commands:
  start    Launch all hyperparameter searches in parallel (default)
  status   Show status of running/completed searches
  summary  Print best configs found from completed searches
  kill     Kill all tracked search processes
EOF
    exit 0
}

start_searches() {
    mkdir -p "$LOG_DIR"
    > "$PID_FILE"

    echo "=================================================="
    echo " FedQHD Hyperparameter Search Launcher"
    echo "=================================================="
    echo " Environments : ${ENVS[*]}"
    echo " Method types : ${TYPES[*]}"
    echo " Episodes     : $EPISODES"
    echo " Agents       : $AGENT_NUM"
    echo " Runs/combo   : $RUNS"
    echo "=================================================="
    echo ""

    for ENV in "${ENVS[@]}"; do
        for TYPE in "${TYPES[@]}"; do
            LOG="$LOG_DIR/hparam_${ENV}_${TYPE}.log"
            CMD="$PYTHON $SCRIPT_DIR/hparam_search.py \
                --method_type $TYPE \
                --env $ENV \
                --episodes $EPISODES \
                --agent_num $AGENT_NUM \
                --runs $RUNS"
            nohup $CMD > "$LOG" 2>&1 &
            PID=$!
            echo "$PID $ENV $TYPE $LOG" >> "$PID_FILE"
            echo "  [PID $PID] $ENV / $TYPE  →  $LOG"
        done
    done

    echo ""
    echo "All searches launched. Monitor with:"
    echo "  $0 status"
    echo "  $0 summary   (once complete)"
}

status_searches() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found. Run '$0 start' first."
        exit 1
    fi

    printf "%-8s %-14s %-6s %-12s %s\n" "PID" "ENV" "TYPE" "STATUS" "LOG"
    printf "%-8s %-14s %-6s %-12s %s\n" "---" "---" "----" "------" "---"
    while read -r PID ENV TYPE LOG; do
        if kill -0 "$PID" 2>/dev/null; then
            # Show progress: count completed combos
            DONE=$(grep -c "mean reward:" "$LOG" 2>/dev/null || echo 0)
            STATUS="RUNNING ($DONE done)"
        else
            if grep -q "Results saved to" "$LOG" 2>/dev/null; then
                STATUS="DONE"
            else
                STATUS="FAILED/EXIT"
            fi
        fi
        printf "%-8s %-14s %-6s %-12s %s\n" "$PID" "$ENV" "$TYPE" "$STATUS" "$LOG"
    done < "$PID_FILE"
}

summary_searches() {
    RESULT_DIR="$SCRIPT_DIR/results/hparam_search"
    if [[ ! -d "$RESULT_DIR" ]]; then
        echo "No results yet in $RESULT_DIR"
        exit 1
    fi

    echo ""
    echo "================================================================"
    echo " HYPERPARAMETER SEARCH SUMMARY"
    echo "================================================================"
    for JSON in "$RESULT_DIR"/*.json; do
        [[ -f "$JSON" ]] || continue
        echo ""
        echo "--- $(basename "$JSON") ---"
        python3 - "$JSON" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
results = data["results"]
seen = set()
print(f"  {'Method':<22} {'Best LR':<9} {'Best Agg':<10} {'Reward':>10}")
print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*10}")
for r in results:
    if r["method"] not in seen:
        seen.add(r["method"])
        print(f"  {r['method']:<22} {r['learning_rate']:<9} "
              f"{r['aggregation_interval']:<10} {r['mean_reward']:>10.2f}")
PYEOF
    done
    echo ""
    echo "================================================================"
}

kill_searches() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found."
        exit 1
    fi
    while read -r PID ENV TYPE _LOG; do
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "  Killed PID $PID ($ENV / $TYPE)"
        else
            echo "  PID $PID ($ENV / $TYPE) already stopped"
        fi
    done < "$PID_FILE"
}

COMMAND="${1:-start}"
case "$COMMAND" in
    start)   start_searches ;;
    status)  status_searches ;;
    summary) summary_searches ;;
    kill)    kill_searches ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $COMMAND"; usage ;;
esac
