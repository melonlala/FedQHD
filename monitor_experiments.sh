#!/bin/bash

# Monitor the three parallel experiments

echo "======================================"
echo "FedQHD Experiments Monitor"
echo "======================================"
echo ""

# Check if processes are running
echo "Running Processes:"
ps aux | grep "run_experiments.py" | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15}'
echo ""

# Check Q1 progress
echo "--- Q1 (Homogeneous Encoders) ---"
if [ -f q1_output.log ]; then
    tail -5 q1_output.log | grep -E "(RUN|Training|%|completed|Saved)"
else
    echo "Q1 log not found"
fi
echo ""

# Check Q2 progress
echo "--- Q2 (Heterogeneous Encoders) ---"
if [ -f q2_output.log ]; then
    tail -5 q2_output.log | grep -E "(RUN|Training|%|completed|Saved)"
else
    echo "Q2 log not found"
fi
echo ""

# Check Scalability progress
echo "--- Scalability ---"
if [ -f scalability_output.log ]; then
    tail -5 scalability_output.log | grep -E "(N =|Training|%|completed|Saved)"
else
    echo "Scalability log not found"
fi
echo ""

# Check file sizes
echo "Log File Sizes:"
ls -lh *_output.log 2>/dev/null | awk '{print $9, $5}'
echo ""

# Check results directory
echo "Results Generated:"
if [ -d results/CartPole ]; then
    find results/CartPole -name "*.pdf" -o -name "*.json" -o -name "*.tex" | head -20
else
    echo "No results yet"
fi
