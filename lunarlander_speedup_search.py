#!/usr/bin/env python3
"""
Hyperparameter search for FedQHD on LunarLander, optimised for showing
*learning speedup* rather than just final performance.

Metrics reported per combo:
  - auc_ratio      : sum(FedQHD rewards) / sum(Independent rewards)
                     > 1 means FedQHD accumulated more reward over training
  - speedup_ep     : Independent convergence_ep - FedQHD convergence_ep
                     > 0 means FedQHD converged earlier (in episodes)
  - final_gap      : FedQHD final_avg_reward - Independent final_avg_reward

Usage:
    python lunarlander_speedup_search.py              # uses all defaults
    python lunarlander_speedup_search.py --episodes 500 --runs 2 --agent_num 5
    python lunarlander_speedup_search.py --quick      # fast sanity check (200 eps, 1 run)
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import train_independent_qhd, train_fedqhd_homogeneous

# ── Search grid ────────────────────────────────────────────────────────────────
# Focused grid: 16 combos × ~3 min/combo ≈ ~50 min total.
# max_episode_steps=400 prevents runaway long episodes as agent improves.
SEARCH_GRID = {
    "learning_rate":        [0.05, 0.1, 0.2],
    "exploration_decay":    [0.99, 0.999],
    "aggregation_interval": [10, 25],
    "rff_gamma":            [0.5, 1.0],
}

# Fixed hyperparameters (not searched)
FIXED = dict(
    env                = "LunarLander",
    agent_num          = 3,           # 3 agents: clear federation benefit, manageable speed
    hyperdimension     = 10000,
    discount_factor    = 0.99,        # LunarLander is long-horizon
    exploration_rate   = 1.0,
    exploration_min    = 0.01,
    anchor_set_size    = 200,
    rff_gamma          = 1.0,         # overridden per combo
    grid_size          = [8, 8],
    test_heterogeneous = False,
    max_episode_steps  = 400,         # cap episode length to prevent slowdown as agent improves
)

# Reward threshold used to measure "convergence speed"
# LunarLander: 0 = any landing; 100 = consistent soft landing
CONVERGENCE_THRESHOLD = 0.0
# ──────────────────────────────────────────────────────────────────────────────


class SimpleArgs:
    def __init__(self, **kwargs):
        for k, v in FIXED.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)


def episodes_to_threshold(reward_history, threshold, window=20):
    """
    Return the first episode where the rolling mean (window) exceeds `threshold`.
    Returns None if never reached.
    """
    for i in range(window, len(reward_history) + 1):
        if np.mean(reward_history[i - window: i]) >= threshold:
            return i
    return None


def compute_metrics(ind_result, fed_result, threshold=CONVERGENCE_THRESHOLD):
    """Return dict of speedup-oriented metrics."""
    ind_r = ind_result.reward_history
    fed_r = fed_result.reward_history

    n = max(len(ind_r), len(fed_r))
    # Pad shorter history with its last value for fair AUC comparison
    ind_padded = ind_r + [ind_r[-1]] * (n - len(ind_r)) if ind_r else [0] * n
    fed_padded = fed_r + [fed_r[-1]] * (n - len(fed_r)) if fed_r else [0] * n

    ind_auc = float(np.sum(ind_padded))
    fed_auc = float(np.sum(fed_padded))

    # auc_improvement: positive means FedQHD accumulated more total reward.
    # Works correctly even when rewards are negative (LunarLander baseline scenario).
    auc_improvement = fed_auc - ind_auc

    # Normalised ratio: positive = Fed better, scale-free.
    # Use absolute ind_auc as denominator so sign is always informative.
    auc_ratio = auc_improvement / (abs(ind_auc) + 1e-9)

    ind_conv = episodes_to_threshold(ind_r, threshold)
    fed_conv = episodes_to_threshold(fed_r, threshold)

    if ind_conv is not None and fed_conv is not None:
        speedup_ep = ind_conv - fed_conv
    elif ind_conv is None and fed_conv is not None:
        speedup_ep = float("inf")   # FedQHD solved it, Independent never did
    else:
        speedup_ep = None  # neither converged

    return {
        "ind_final":  ind_result.final_avg_reward,
        "fed_final":  fed_result.final_avg_reward,
        "final_gap":  fed_result.final_avg_reward - ind_result.final_avg_reward,
        "ind_auc":        ind_auc,
        "fed_auc":        fed_auc,
        "auc_improvement": auc_improvement,
        "auc_ratio":      auc_ratio,
        "ind_conv_ep": ind_conv,
        "fed_conv_ep": fed_conv,
        "speedup_ep":  speedup_ep,
        "ind_time":   ind_result.training_time,
        "fed_time":   fed_result.training_time,
    }


def run_combo(episodes, lr, eps_decay, agg, gamma, agent_num, runs, seed_base=42):
    """Run one hyperparameter combo; average metrics over `runs` seeds."""
    all_metrics = []

    for run in range(runs):
        seed = seed_base + run * 100
        np.random.seed(seed)

        args = SimpleArgs(
            learning_rate        = lr,
            exploration_decay    = eps_decay,
            aggregation_interval = agg,
            rff_gamma            = gamma,
            agent_num            = agent_num,
            random_seed          = seed,
        )
        args.episodes = episodes  # handy reference but train_* take episodes directly

        try:
            ind_res = train_independent_qhd(episodes, args)
        except Exception as e:
            print(f"    [run {run}] Independent ERROR: {e}")
            continue
        try:
            fed_res = train_fedqhd_homogeneous(episodes, args)
        except Exception as e:
            print(f"    [run {run}] FedQHD ERROR: {e}")
            continue

        m = compute_metrics(ind_res, fed_res)
        all_metrics.append(m)
        print(
            f"    run {run}: ind={m['ind_final']:+.1f}  fed={m['fed_final']:+.1f}  "
            f"gap={m['final_gap']:+.1f}  auc_gain={m['auc_improvement']:+.0f}  "
            f"speedup_ep={m['speedup_ep']}"
        )

    if not all_metrics:
        return None

    # Average scalar metrics across runs
    averaged = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics
                if m[key] is not None and m[key] != float("inf")
                and not (isinstance(m[key], float) and np.isinf(m[key]))]
        averaged[key] = float(np.mean(vals)) if vals else None

    # Keep track of how often FedQHD solved it when Independent did not
    n_inf_speedup = sum(1 for m in all_metrics if m["speedup_ep"] == float("inf"))
    averaged["n_inf_speedup"] = n_inf_speedup
    averaged["n_runs"]        = len(all_metrics)

    return averaged


def run_full_search(episodes, agent_num, runs, quick=False, output_dir="results/lunarlander_speedup"):
    """Iterate over all combos and rank by AUC ratio."""

    lrs       = SEARCH_GRID["learning_rate"]
    eps_decays = SEARCH_GRID["exploration_decay"]
    aggs      = SEARCH_GRID["aggregation_interval"]
    gammas    = SEARCH_GRID["rff_gamma"]

    if quick:
        lrs        = [0.1]
        eps_decays = [0.995]
        aggs       = [10, 25]
        gammas     = [1.0]

    total    = len(lrs) * len(eps_decays) * len(aggs) * len(gammas)
    done     = 0
    all_rows = []

    print(f"\n{'='*70}")
    print(f"LunarLander FedQHD Speedup Search")
    print(f"{'='*70}")
    print(f"  episodes  : {episodes}")
    print(f"  agent_num : {agent_num}")
    print(f"  runs/combo: {runs}")
    print(f"  total combos: {total}")
    print(f"  threshold for conv. speed: {CONVERGENCE_THRESHOLD}")
    print(f"{'='*70}\n")

    t0 = time.time()

    for lr in lrs:
        for eps_decay in eps_decays:
            for agg in aggs:
                for gamma in gammas:
                    done += 1
                    tag = (f"[{done}/{total}]  lr={lr}  eps_decay={eps_decay}  "
                           f"agg={agg}  rff_gamma={gamma}")
                    print(f"\n{'-'*70}\n{tag}")

                    metrics = run_combo(episodes, lr, eps_decay, agg, gamma,
                                        agent_num, runs)

                    if metrics is None:
                        print("  SKIPPED (all runs failed)")
                        continue

                    row = {
                        "learning_rate":        lr,
                        "exploration_decay":    eps_decay,
                        "aggregation_interval": agg,
                        "rff_gamma":            gamma,
                        **metrics,
                    }
                    all_rows.append(row)

                    elapsed = time.time() - t0
                    print(
                        f"  → auc_gain={metrics['auc_improvement']:+.0f}  "
                        f"auc_ratio={metrics['auc_ratio']:+.3f}  "
                        f"final_gap={metrics['final_gap']:+.1f}  "
                        f"speedup_ep={metrics['speedup_ep']}  "
                        f"elapsed={elapsed:.0f}s"
                    )

    # Sort by auc_improvement descending (primary = most cumulative gain),
    # then final_gap (secondary = best endpoint).
    all_rows.sort(key=lambda r: (
        r.get("auc_improvement") or 0,
        r.get("final_gap") or 0
    ), reverse=True)

    _print_summary(all_rows)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "LunarLander_speedup_search.json")
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "episodes":    episodes,
                "agent_num":   agent_num,
                "runs":        runs,
                "threshold":   CONVERGENCE_THRESHOLD,
                "fixed":       {k: v for k, v in FIXED.items() if k not in
                                ("rff_gamma",)},  # rff_gamma is searched
            },
            "results": all_rows,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return all_rows


def _print_summary(rows, top_n=10):
    print(f"\n{'='*70}")
    print("  TOP COMBOS BY AUC GAIN  (FedQHD cumulative reward - Independent)")
    print(f"{'='*70}")
    hdr = (f"{'Rank':<5} {'LR':<6} {'eps_dec':<8} {'agg':<5} {'gamma':<7} "
           f"{'auc_gain':>10} {'gap':>7} {'speedup_ep':>11}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows[:top_n], 1):
        sp = r.get("speedup_ep")
        sp_str = f"{sp:+.0f}" if sp is not None else "N/A"
        print(
            f"{i:<5} {r['learning_rate']:<6} {r['exploration_decay']:<8} "
            f"{r['aggregation_interval']:<5} {r['rff_gamma']:<7} "
            f"{(r.get('auc_improvement') or float('nan')):>+10.0f} "
            f"{(r.get('final_gap') or float('nan')):>+7.1f} "
            f"{sp_str:>11}"
        )
    print()

    # Best per aggregation interval
    print("  BEST CONFIG PER AGGREGATION INTERVAL:")
    seen_agg = {}
    for r in rows:
        a = r["aggregation_interval"]
        if a not in seen_agg:
            seen_agg[a] = r
    for a, r in sorted(seen_agg.items()):
        print(
            f"  agg={a:<4}  lr={r['learning_rate']}  eps_dec={r['exploration_decay']}  "
            f"gamma={r['rff_gamma']}  →  auc_gain={r.get('auc_improvement'):+.0f}  "
            f"gap={r.get('final_gap'):+.1f}"
        )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FedQHD speedup search for LunarLander"
    )
    parser.add_argument("--episodes",   type=int,   default=200,
                        help="Episodes per run (default 200)")
    parser.add_argument("--agent_num",  type=int,   default=3,
                        help="Number of federated agents (default 3)")
    parser.add_argument("--runs",       type=int,   default=1,
                        help="Independent seeds per combo (default 1)")
    parser.add_argument("--output_dir", default="results/lunarlander_speedup")
    parser.add_argument("--quick",      action="store_true",
                        help="Fast sanity check with minimal grid")
    args = parser.parse_args()

    run_full_search(
        episodes   = args.episodes,
        agent_num  = args.agent_num,
        runs       = args.runs,
        quick      = args.quick,
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
