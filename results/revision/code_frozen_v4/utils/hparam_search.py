#!/usr/bin/env python3
"""
Hyperparameter grid search for QHD and DQN methods.

Usage:
    python hparam_search.py --method_type qhd --env CartPole --episodes 500
    python hparam_search.py --method_type dqn --env CartPole --episodes 500

    # LunarLander: search larger LRs and rff_gamma
    python hparam_search.py --method_type qhd --env LunarLander --episodes 500 \
        --lr_values "0.1,0.5,1.0,5.0,10.0" --rff_gammas "0.5,1.0,2.0,5.0"
"""

import argparse
import copy
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import (
    train_independent_qhd,
    train_fedqhd_homogeneous,
    train_fedqhd_heterogeneous,
    train_oracle_qhd,
    train_fedavg_dqn,
    train_oracle_dqn,
)

# ── Default hyperparameter grids ──────────────────────────────────────────────

QHD_GRID = {
    "learning_rate":        [0.001, 0.01, 0.05, 0.1],
    "aggregation_interval": [10, 25, 50, 100],
}

DQN_GRID = {
    "learning_rate":        [0.0001, 0.001, 0.005, 0.01],
    "aggregation_interval": [10, 25, 50, 100],
}

# Heterogeneous encoder grid: wider LR range because Oracle auto-divides by N.
# With N agents, oracle_lr = qhd_lr / N, so lr=0.5 → effective_lr=0.1 for N=5.
# exploration_decay is a search axis here (excluded from QHD/DQN grids for speed).
HETERO_GRID = {
    "learning_rate":        [0.1, 0.2, 0.5],
    "aggregation_interval": [10, 25],
    "exploration_decay":    [0.990, 0.995, 0.999],
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

HETERO_METHODS = {
    "FedQHD (Hetero)":     train_fedqhd_heterogeneous,
    "Oracle QHD (Hetero)": lambda ep, args: train_oracle_qhd(ep, args, use_heterogeneous=True),
}

# ──────────────────────────────────────────────────────────────────────────────

def make_args(base_args, lr, agg_interval, rff_gamma=None, exploration_decay=None):
    """Return a copy of args with overridden lr, agg_interval, and optional fields.

    Sets both the generic `learning_rate` / `aggregation_interval` fields AND the
    per-method-type fields (`qhd_lr`, `qhd_agg_interval`, `dqn_lr`,
    `dqn_agg_interval`) that experiments_runner.py reads directly.
    """
    a = copy.copy(base_args)
    a.learning_rate = lr
    a.aggregation_interval = agg_interval
    # Per-type fields consumed by experiments_runner.py training functions
    a.qhd_lr = lr
    a.qhd_agg_interval = agg_interval
    a.dqn_lr = lr
    a.dqn_agg_interval = agg_interval
    if rff_gamma is not None:
        a.rff_gamma = rff_gamma
    if exploration_decay is not None:
        a.exploration_decay = exploration_decay
        a.qhd_exploration_decay = exploration_decay
    return a


def run_grid(method_type: str, base_args, num_runs: int = 2,
             rff_gammas=None, exploration_decays=None):
    """
    Run full grid search for the given method type.

    Args:
        rff_gammas:        list of rff_gamma values to sweep (None → base_args.rff_gamma only).
        exploration_decays: list of epsilon-decay values to sweep (None → grid default or
                            base_args.qhd_exploration_decay only).

    Returns list of result dicts sorted by mean final reward (descending).
    """
    if method_type == "hetero":
        grid    = HETERO_GRID
        methods = HETERO_METHODS
    elif method_type == "qhd":
        grid    = QHD_GRID
        methods = QHD_METHODS
    else:
        grid    = DQN_GRID
        methods = DQN_METHODS

    lrs  = grid["learning_rate"]
    aggs = grid["aggregation_interval"]

    if rff_gammas is None:
        rff_gammas = [base_args.rff_gamma]

    if exploration_decays is None:
        # Use grid-embedded decay list (hetero) or single fixed value (qhd/dqn)
        exploration_decays = grid.get(
            "exploration_decay",
            [getattr(base_args, 'qhd_exploration_decay', base_args.exploration_decay)]
        )

    all_results = []
    total = len(methods) * len(lrs) * len(aggs) * len(rff_gammas) * len(exploration_decays)
    done  = 0

    for rff_gamma in rff_gammas:
        for eps_decay in exploration_decays:
            for method_name, train_fn in methods.items():
                for lr in lrs:
                    for agg in aggs:
                        done += 1
                        gamma_tag = f"  γ={rff_gamma}" if len(rff_gammas) > 1 else ""
                        decay_tag = f"  decay={eps_decay}" if len(exploration_decays) > 1 else ""
                        tag = f"[{done}/{total}] {method_name}  lr={lr}  agg={agg}{gamma_tag}{decay_tag}"
                        print(f"\n{'='*70}\n{tag}\n{'='*70}")

                        run_rewards = []
                        for run in range(num_runs):
                            a = make_args(base_args, lr, agg, rff_gamma,
                                          exploration_decay=eps_decay)
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
                            "method":               method_name,
                            "learning_rate":        lr,
                            "aggregation_interval": agg,
                            "rff_gamma":            rff_gamma,
                            "exploration_decay":    eps_decay,
                            "mean_reward":          mean_r,
                            "std_reward":           std_r,
                            "num_runs":             len(run_rewards),
                        })

    all_results.sort(key=lambda x: x["mean_reward"], reverse=True)
    return all_results


def print_summary(results, method_type, multi_gamma=False, multi_decay=False):
    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH RESULTS — {method_type.upper()} METHODS")
    print(f"{'='*70}")

    # Build header dynamically based on which axes vary
    extra = ""
    if multi_gamma:
        extra += f" {'γ':<6}"
    if multi_decay:
        extra += f" {'decay':<7}"

    print(f"{'Rank':<5} {'Method':<22} {'LR':<8} {'Agg':<6}{extra} {'Reward':>10}  {'Std':>8}")
    print(f"{'-'*5} {'-'*22} {'-'*8} {'-'*6}{'-'*len(extra)} {'-'*10}  {'-'*8}")
    for rank, r in enumerate(results, 1):
        extra_vals = ""
        if multi_gamma:
            extra_vals += f" {r['rff_gamma']:<6}"
        if multi_decay:
            extra_vals += f" {r['exploration_decay']:<7}"
        print(
            f"{rank:<5} {r['method']:<22} {r['learning_rate']:<8} "
            f"{r['aggregation_interval']:<6}{extra_vals} "
            f"{r['mean_reward']:>10.2f}  {r['std_reward']:>8.2f}"
        )
    print()

    # Best per method
    seen = set()
    print("  BEST CONFIG PER METHOD:")
    for r in results:
        if r["method"] not in seen:
            seen.add(r["method"])
            gamma_str = f",  γ={r['rff_gamma']}" if multi_gamma else ""
            decay_str = f",  decay={r['exploration_decay']}" if multi_decay else ""
            print(
                f"  {r['method']}: lr={r['learning_rate']},  "
                f"agg_interval={r['aggregation_interval']}{gamma_str}{decay_str},  "
                f"reward={r['mean_reward']:.2f}"
            )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for FedQHD")
    parser.add_argument("--method_type", required=True, choices=["qhd", "dqn", "hetero"],
                        help="Which family of methods to search: "
                             "qhd (homo: IndQHD/FedQHD/OracleQHD), "
                             "dqn (FedAvg-DQN/OracleDQN), "
                             "hetero (FedQHD-Hetero/OracleQHD-Hetero + decay sweep)")
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
    parser.add_argument("--rff_gamma",         type=float, default=1.0,
                        help="Default rff_gamma (used when --rff_gammas is not set)")
    parser.add_argument("--rff_gammas",        type=str,   default=None,
                        help="Comma-separated rff_gamma values to search, e.g. '0.5,1.0,2.0,5.0'")
    parser.add_argument("--lr_values",         type=str,   default=None,
                        help="Comma-separated learning rates (overrides default grid), "
                             "e.g. '0.1,0.5,1.0,5.0,10.0'")
    parser.add_argument("--agg_values",        type=str,   default=None,
                        help="Comma-separated aggregation intervals (overrides default grid), "
                             "e.g. '10,25,50'")
    parser.add_argument("--eps_decay_values",  type=str,   default=None,
                        help="Comma-separated exploration_decay values to search (overrides grid), "
                             "e.g. '0.990,0.995,0.999'")
    parser.add_argument("--anchor_set_size",   type=int,   default=200)
    parser.add_argument("--grid_size",         type=int,   nargs=2, default=[8, 8])
    parser.add_argument("--output_dir", default="results/hparam_search")
    args = parser.parse_args()

    # Placeholders consumed by training functions but overridden per combo
    args.learning_rate        = 0.01
    args.aggregation_interval = 50
    args.random_seed          = 42

    # Per-type hyperparameter fields needed by experiments_runner.py
    args.qhd_lr               = args.learning_rate
    args.qhd_agg_interval     = args.aggregation_interval
    args.qhd_discount         = args.discount_factor
    args.qhd_exploration_rate  = args.exploration_rate
    args.qhd_exploration_decay = args.exploration_decay
    args.qhd_exploration_min   = args.exploration_min
    args.dqn_lr               = 0.001
    args.dqn_agg_interval     = 10

    # Override grid with CLI values if provided
    if args.lr_values is not None:
        lr_list = [float(x.strip()) for x in args.lr_values.split(",")]
        if args.method_type == "qhd":
            QHD_GRID["learning_rate"] = lr_list
        elif args.method_type == "hetero":
            HETERO_GRID["learning_rate"] = lr_list
        else:
            DQN_GRID["learning_rate"] = lr_list
        print(f"Custom LR grid: {lr_list}")

    if args.agg_values is not None:
        agg_list = [int(x.strip()) for x in args.agg_values.split(",")]
        if args.method_type == "qhd":
            QHD_GRID["aggregation_interval"] = agg_list
        elif args.method_type == "hetero":
            HETERO_GRID["aggregation_interval"] = agg_list
        else:
            DQN_GRID["aggregation_interval"] = agg_list
        print(f"Custom agg grid: {agg_list}")

    # Parse rff_gammas
    rff_gammas = None
    if args.rff_gammas is not None:
        rff_gammas = [float(x.strip()) for x in args.rff_gammas.split(",")]
        print(f"rff_gamma search: {rff_gammas}")

    # Parse exploration_decay values
    exploration_decays = None
    if args.eps_decay_values is not None:
        exploration_decays = [float(x.strip()) for x in args.eps_decay_values.split(",")]
        # Also override the grid's embedded decay list
        if args.method_type == "hetero":
            HETERO_GRID["exploration_decay"] = exploration_decays
        print(f"exploration_decay search: {exploration_decays}")

    multi_gamma = rff_gammas is not None and len(rff_gammas) > 1
    multi_decay = (exploration_decays is not None and len(exploration_decays) > 1) or \
                  (args.method_type == "hetero" and len(HETERO_GRID["exploration_decay"]) > 1)

    print(f"\nHyperparameter search: {args.method_type.upper()} methods")
    print(f"Environment : {args.env}")
    print(f"Episodes    : {args.episodes}")
    print(f"Agents      : {args.agent_num}")
    print(f"Runs/combo  : {args.runs}")
    print(f"Discount    : {args.qhd_discount}")
    print(f"Eps decay   : {args.qhd_exploration_decay}  (searched: {exploration_decays or 'fixed'})")
    if args.method_type == "hetero":
        grid    = HETERO_GRID
        methods = HETERO_METHODS
        print(f"Note        : Oracle QHD (Hetero) uses effective_lr = qhd_lr / {args.agent_num} "
              f"(auto-scaled inside train_oracle_qhd)")
    elif args.method_type == "qhd":
        grid    = QHD_GRID
        methods = QHD_METHODS
    else:
        grid    = DQN_GRID
        methods = DQN_METHODS
    n_combos = len(grid["learning_rate"]) * len(grid["aggregation_interval"])
    n_gammas  = len(rff_gammas) if rff_gammas else 1
    n_decays  = len(exploration_decays) if exploration_decays else len(grid.get("exploration_decay", [1]))
    print(f"Combos      : {len(methods)} methods × {n_combos} LR/agg combos × "
          f"{n_gammas} γ × {n_decays} decay = "
          f"{len(methods) * n_combos * n_gammas * n_decays} total\n")

    results = run_grid(args.method_type, args, num_runs=args.runs,
                       rff_gammas=rff_gammas, exploration_decays=exploration_decays)
    print_summary(results, args.method_type, multi_gamma=multi_gamma, multi_decay=multi_decay)

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
