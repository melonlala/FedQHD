#!/usr/bin/env python3
"""
Hyperparameter grid search for QHD and DQN methods.

Usage:
    python hparam_search.py --method_type qhd --env CartPole --episodes 500
    python hparam_search.py --method_type dqn --env CartPole --episodes 500
"""

import argparse
import copy
import json
import os
import sys
import types
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import (
    train_independent_qhd,
    train_fedqhd_homogeneous,
    train_oracle_qhd,
    train_fedavg_dqn,
    train_oracle_dqn,
)

# ── Hyperparameter grids ───────────────────────────────────────────────────────

QHD_GRID = {
    "learning_rate":        [0.001, 0.01, 0.05, 0.1],
    "aggregation_interval": [10, 25, 50, 100],
}

DQN_GRID = {
    "learning_rate":        [0.0001, 0.001, 0.005, 0.01],
    "aggregation_interval": [10, 25, 50, 100],
}

# Methods tested for each type
QHD_METHODS = {
    "Independent QHD":    train_independent_qhd,
    "FedQHD (Homo)":      train_fedqhd_homogeneous,
    "Oracle QHD":         lambda ep, args: train_oracle_qhd(ep, args, use_heterogeneous=False),
}

DQN_METHODS = {
    "FedAvg-DQN":  train_fedavg_dqn,
    "Oracle DQN":  train_oracle_dqn,
}

# ──────────────────────────────────────────────────────────────────────────────

def make_args(base_args, lr, agg_interval):
    """Return a copy of args with overridden lr and agg_interval."""
    a = copy.copy(base_args)
    a.learning_rate = lr
    a.aggregation_interval = agg_interval
    return a


def run_grid(method_type: str, base_args, num_runs: int = 2):
    """
    Run full grid search for the given method type.
    Returns list of result dicts sorted by mean final reward (descending).
    """
    grid   = QHD_GRID   if method_type == "qhd" else DQN_GRID
    methods = QHD_METHODS if method_type == "qhd" else DQN_METHODS

    lrs  = grid["learning_rate"]
    aggs = grid["aggregation_interval"]

    all_results = []
    total = len(methods) * len(lrs) * len(aggs)
    done  = 0

    for method_name, train_fn in methods.items():
        for lr in lrs:
            for agg in aggs:
                done += 1
                tag = f"[{done}/{total}] {method_name}  lr={lr}  agg={agg}"
                print(f"\n{'='*70}\n{tag}\n{'='*70}")

                run_rewards = []
                for run in range(num_runs):
                    a = make_args(base_args, lr, agg)
                    a.random_seed = 42 + run
                    np.random.seed(a.random_seed)
                    try:
                        res = train_fn(base_args.episodes, a)
                        run_rewards.append(res.final_avg_reward)
                    except Exception as e:
                        print(f"  ERROR run {run}: {e}")

                if run_rewards:
                    mean_r = float(np.mean(run_rewards))
                    std_r  = float(np.std(run_rewards))
                else:
                    mean_r, std_r = float("nan"), float("nan")

                print(f"  → mean reward: {mean_r:.2f} ± {std_r:.2f}")
                all_results.append({
                    "method":              method_name,
                    "learning_rate":       lr,
                    "aggregation_interval": agg,
                    "mean_reward":         mean_r,
                    "std_reward":          std_r,
                    "num_runs":            len(run_rewards),
                })

    all_results.sort(key=lambda x: x["mean_reward"], reverse=True)
    return all_results


def print_summary(results, method_type):
    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH RESULTS — {method_type.upper()} METHODS")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Method':<22} {'LR':<8} {'Agg':<6} {'Reward':>10}  {'Std':>8}")
    print(f"{'-'*5} {'-'*22} {'-'*8} {'-'*6} {'-'*10}  {'-'*8}")
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {r['method']:<22} {r['learning_rate']:<8} "
            f"{r['aggregation_interval']:<6} {r['mean_reward']:>10.2f}  {r['std_reward']:>8.2f}"
        )
    print()

    # Best per method
    seen = set()
    print("  BEST CONFIG PER METHOD:")
    for r in results:
        if r["method"] not in seen:
            seen.add(r["method"])
            print(
                f"  {r['method']}: lr={r['learning_rate']},  "
                f"agg_interval={r['aggregation_interval']},  "
                f"reward={r['mean_reward']:.2f}"
            )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for FedQHD")
    parser.add_argument("--method_type", required=True, choices=["qhd", "dqn"],
                        help="Which family of methods to search")
    parser.add_argument("--env", default="CartPole",
                        choices=["CartPole", "Acrobot", "LunarLander",
                                 "MountainCar", "Taxi", "CliffWalking"])
    parser.add_argument("--episodes",    type=int,   default=500)
    parser.add_argument("--agent_num",   type=int,   default=3)
    parser.add_argument("--runs",        type=int,   default=2,
                        help="Independent runs per combo (averaged)")
    parser.add_argument("--hyperdimension",    type=int,   default=10000)
    parser.add_argument("--discount_factor",   type=float, default=0.99)
    parser.add_argument("--exploration_rate",  type=float, default=1.0)
    parser.add_argument("--exploration_decay", type=float, default=0.995)
    parser.add_argument("--exploration_min",   type=float, default=0.01)
    parser.add_argument("--rff_gamma",         type=float, default=1.0)
    parser.add_argument("--anchor_set_size",   type=int,   default=200)
    parser.add_argument("--grid_size",         type=int,   nargs=2, default=[8, 8])
    parser.add_argument("--output_dir", default="results/hparam_search")
    args = parser.parse_args()

    # Placeholders consumed by training functions but overridden per combo
    args.learning_rate        = 0.01
    args.aggregation_interval = 50
    args.random_seed          = 42

    print(f"\nHyperparameter search: {args.method_type.upper()} methods")
    print(f"Environment : {args.env}")
    print(f"Episodes    : {args.episodes}")
    print(f"Agents      : {args.agent_num}")
    print(f"Runs/combo  : {args.runs}")
    grid = QHD_GRID if args.method_type == "qhd" else DQN_GRID
    n_combos = len(grid["learning_rate"]) * len(grid["aggregation_interval"])
    methods  = QHD_METHODS if args.method_type == "qhd" else DQN_METHODS
    print(f"Combos      : {len(methods)} methods × {n_combos} hyperparameter combos = "
          f"{len(methods) * n_combos} total\n")

    results = run_grid(args.method_type, args, num_runs=args.runs)
    print_summary(results, args.method_type)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir, f"{args.env}_{args.method_type}_hparam_search.json"
    )
    with open(out_file, "w") as f:
        json.dump({"env": args.env, "method_type": args.method_type,
                   "results": results}, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
