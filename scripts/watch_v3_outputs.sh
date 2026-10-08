#!/usr/bin/env bash
# Regenerate the main_v3 tables/figures as soon as every (method, cell) has the seed set:
#   seeds 42-44 -> results/revision/v3_3seed   (then)   seeds 42-46 -> results/revision/v3_5seed
# A READY file is written next to the outputs when a seed set is complete.
set -u
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$W
for SEEDS in 42,43,44 42,43,44,45,46; do
  N=$(echo "$SEEDS" | tr ',' '\n' | wc -l)
  OUT=$W/results/revision/v3_${N}seed
  while true; do
    msg=$(python "$W/scripts/make_v3_outputs.py" --seeds "$SEEDS" --out "$OUT" 2>&1 | tail -1)
    echo "[$(date '+%F %T')] $SEEDS: $msg"
    if echo "$msg" | grep -q "cells with missing seeds: 0/"; then
      date '+%F %T' > "$OUT/READY"; break
    fi
    sleep 600
  done
done
