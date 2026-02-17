#!/bin/bash

echo "=========================================="
echo "FedQHD Experiments - Status Overview"
echo "=========================================="
echo ""

# Count running processes
RUNNING=$(ps aux | grep "run_experiments.py" | grep -v grep | wc -l)
echo "Running experiments: $RUNNING / 11"
echo ""

echo "=========================================="
echo "CARTPOLE"
echo "=========================================="
echo "Q1:" && tail -n 2 q1_output.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Q2:" && tail -n 2 q2_output.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Scalability:" && tail -n 2 scalability_output.log 2>/dev/null | grep -E "(%|N =|completed|Saved)" | tail -n 1
echo ""

echo "=========================================="
echo "ACROBOT"
echo "=========================================="
echo "Q1:" && tail -n 2 acrobot_q1.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Q2:" && tail -n 2 acrobot_q2.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Scalability:" && tail -n 2 acrobot_scalability.log 2>/dev/null | grep -E "(%|N =|completed|Saved)" | tail -n 1
echo ""

echo "=========================================="
echo "LUNARLANDER"
echo "=========================================="
echo "Q1:" && tail -n 2 lunarlander_q1.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Q2:" && tail -n 2 lunarlander_q2.log 2>/dev/null | grep -E "(%|RUN|completed|Saved)" | tail -n 1
echo "Scalability:" && tail -n 2 lunarlander_scalability.log 2>/dev/null | grep -E "(%|N =|completed|Saved)" | tail -n 1
echo ""

echo "=========================================="
echo "MOUNTAINCAR"
echo "=========================================="
echo "Scalability:" && tail -n 2 mountaincar_scalability.log 2>/dev/null | grep -E "(%|N =|completed|Saved)" | tail -n 1
echo ""

echo "=========================================="
echo "TAXI"
echo "=========================================="
echo "Scalability:" && tail -n 2 taxi_scalability.log 2>/dev/null | grep -E "(%|N =|completed|Saved)" | tail -n 1
echo ""

echo "=========================================="
echo "LOG FILE SIZES"
echo "=========================================="
ls -lh *.log 2>/dev/null | awk '{print $9 ": " $5}'
echo ""

echo "=========================================="
echo "RESULTS DIRECTORIES"
echo "=========================================="
ls -d results/*/ 2>/dev/null | while read dir; do
    echo "$dir: $(find $dir -type f | wc -l) files"
done
