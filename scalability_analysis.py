"""
Scalability Analysis for FedQHD:
  1. Grid search over (learning_rate, aggregation_interval) to find the
     configuration that maximises learning *speedup* vs. Independent QHD.
  2. Full scalability run with the best config across N ∈ {2,5,10,20}.
  3. Single combined learning-curve plot with all N on one axis.

Run:
    python scalability_analysis.py --search          # step 1 only
    python scalability_analysis.py --plot_existing   # step 3 from saved data
    python scalability_analysis.py                   # full pipeline (1→2→3)
"""

import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments_runner import (
    ExperimentResults,
    check_convergence,
    train_fedqhd_homogeneous,
    train_independent_qhd,
)
from env.bounds import state_bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(arr: List[float], window: int = 20) -> np.ndarray:
    """Centered moving average with edge padding."""
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def episodes_to_threshold(
    reward_history: List[float],
    threshold: float,
    window: int = 20,
) -> Optional[int]:
    """
    Return the first episode index where the smoothed reward exceeds
    `threshold`.  Returns None if the threshold is never reached.
    """
    smoothed = smooth(reward_history, window)
    above = np.where(smoothed >= threshold)[0]
    return int(above[0]) if len(above) > 0 else None


def auc_score(reward_history: List[float], window: int = 20) -> float:
    """Area under the smoothed learning curve (higher = faster + better)."""
    return float(np.mean(smooth(reward_history, window)))


def speedup_ratio(
    fed_history: List[float],
    ind_history: List[float],
    threshold: float,
    window: int = 20,
) -> float:
    """
    Convergence speedup = (episodes for Independent to reach threshold) /
                          (episodes for FedQHD to reach threshold).
    Returns 0 if FedQHD never reaches threshold.
    Returns inf if Independent never reaches threshold but FedQHD does.
    """
    fed_ep = episodes_to_threshold(fed_history, threshold, window)
    ind_ep = episodes_to_threshold(ind_history, threshold, window)

    if fed_ep is None:
        return 0.0
    if ind_ep is None:
        return float("inf")
    return ind_ep / fed_ep


# ---------------------------------------------------------------------------
# Shared args factory
# ---------------------------------------------------------------------------

def make_args(
    env: str = "CartPole",
    episodes: int = 300,
    agent_num: int = 10,
    learning_rate: float = 0.1,       # match QHDAgent default (was 0.01 → too slow)
    aggregation_interval: int = 10,
    discount_factor: float = 0.99,
    exploration_rate: float = 1.0,
    exploration_decay: float = 0.99,  # fast decay: reaches ε_min ~460 eps (was 0.9995 → ε≈0.78 at ep500)
    exploration_min: float = 0.01,
    hyperdimension: int = 10000,
    rff_gamma: float = 1.0,
    anchor_set_size: int = 50,
):
    class Args:
        pass

    a = Args()
    a.env = env
    a.episodes = episodes
    a.agent_num = agent_num
    a.learning_rate = learning_rate
    a.aggregation_interval = aggregation_interval
    a.discount_factor = discount_factor
    a.exploration_rate = exploration_rate
    a.exploration_decay = exploration_decay
    a.exploration_min = exploration_min
    a.hyperdimension = hyperdimension
    a.rff_gamma = rff_gamma
    a.anchor_set_size = anchor_set_size
    return a


# ---------------------------------------------------------------------------
# 1. Hyperparameter grid search
# ---------------------------------------------------------------------------

LEARNING_RATES = [0.005, 0.01, 0.05, 0.1]
AGG_INTERVALS = [5, 10, 25, 50]


def run_hyperparam_search(
    env: str = "CartPole",
    search_episodes: int = 300,
    search_agents: int = 10,
    output_dir: str = "results/hyperparam_search",
) -> Tuple[float, int, dict]:
    """
    Grid search over (learning_rate, aggregation_interval).
    Uses AUC + convergence speedup as the selection criterion.

    Returns:
        best_lr, best_agg_interval, full_results_dict
    """
    os.makedirs(output_dir, exist_ok=True)
    results_grid: Dict[str, dict] = {}

    # First get a single Independent baseline for reference
    print("\n" + "=" * 70)
    print(f"Hyperparameter Search  |  env={env}, N={search_agents}, eps={search_episodes}")
    print("=" * 70)

    base_args = make_args(
        env=env,
        episodes=search_episodes,
        agent_num=search_agents,
        learning_rate=0.01,
        aggregation_interval=10,
    )
    print("\n[Baseline] Running Independent QHD for reference...")
    ind_result = train_independent_qhd(search_episodes, base_args)
    ind_history = ind_result.reward_history

    # Threshold = 70% of the independent agent's final average reward
    # (we want FedQHD to reach that *faster*)
    ind_final = np.mean(ind_history[-50:]) if len(ind_history) >= 50 else np.mean(ind_history)
    threshold = max(ind_final * 0.70, np.percentile(ind_history, 60))
    print(f"  Convergence threshold set to {threshold:.2f}")

    best_score = -np.inf
    best_lr, best_agg = 0.01, 10

    total = len(LEARNING_RATES) * len(AGG_INTERVALS)
    run_idx = 0

    for lr in LEARNING_RATES:
        for agg in AGG_INTERVALS:
            run_idx += 1
            key = f"lr={lr}_agg={agg}"
            print(f"\n[{run_idx}/{total}]  lr={lr}  agg_interval={agg}")

            args = make_args(
                env=env,
                episodes=search_episodes,
                agent_num=search_agents,
                learning_rate=lr,
                aggregation_interval=agg,
            )
            fed_result = train_fedqhd_homogeneous(search_episodes, args)
            fed_history = fed_result.reward_history

            auc = auc_score(fed_history)
            spd = speedup_ratio(fed_history, ind_history, threshold)

            # Combined score: normalise AUC + speedup
            # We weight speedup (showing faster convergence) heavily
            score = auc + 20.0 * min(spd, 5.0)  # cap speedup at 5x to avoid inf domination

            print(f"  AUC={auc:.2f}  speedup={spd:.2f}x  score={score:.2f}")

            results_grid[key] = {
                "lr": lr,
                "agg_interval": agg,
                "auc": auc,
                "speedup": spd,
                "score": score,
                "final_reward": fed_result.final_avg_reward,
                "reward_history": fed_history,
            }

            if score > best_score:
                best_score = score
                best_lr, best_agg = lr, agg

    print("\n" + "=" * 70)
    print(f"BEST CONFIG:  lr={best_lr}  agg_interval={best_agg}  (score={best_score:.2f})")
    print("=" * 70)

    # Save
    save_data = {
        "env": env,
        "search_episodes": search_episodes,
        "search_agents": search_agents,
        "threshold": threshold,
        "best_lr": best_lr,
        "best_agg_interval": best_agg,
        "best_score": best_score,
        "grid": results_grid,
        "independent": {
            "auc": auc_score(ind_history),
            "final_reward": ind_result.final_avg_reward,
            "reward_history": ind_history,
        },
    }
    with open(os.path.join(output_dir, "search_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    # Heatmap of scores
    _plot_search_heatmap(results_grid, output_dir)

    return best_lr, best_agg, save_data


def _plot_search_heatmap(results_grid: dict, output_dir: str):
    """Heatmap of combined score over (lr, agg_interval)."""
    lrs = sorted(set(v["lr"] for v in results_grid.values()))
    aggs = sorted(set(v["agg_interval"] for v in results_grid.values()))

    score_mat = np.zeros((len(lrs), len(aggs)))
    auc_mat = np.zeros_like(score_mat)
    spd_mat = np.zeros_like(score_mat)

    for i, lr in enumerate(lrs):
        for j, agg in enumerate(aggs):
            key = f"lr={lr}_agg={agg}"
            if key in results_grid:
                score_mat[i, j] = results_grid[key]["score"]
                auc_mat[i, j] = results_grid[key]["auc"]
                spd_mat[i, j] = min(results_grid[key]["speedup"], 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ["Combined Score", "AUC (avg reward)", "Speedup (×)"]
    mats = [score_mat, auc_mat, spd_mat]

    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(aggs)))
        ax.set_xticklabels(aggs)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels(lrs)
        ax.set_xlabel("Aggregation Interval")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Annotate cells
        for i in range(len(lrs)):
            for j in range(len(aggs)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.suptitle("Hyperparameter Search Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hyperparam_heatmap.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "hyperparam_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_dir}/hyperparam_heatmap.{{pdf,png}}")


# ---------------------------------------------------------------------------
# 2. Full scalability run
# ---------------------------------------------------------------------------

CLIENT_COUNTS = [2, 5, 10, 20]


def run_scalability(
    env: str,
    episodes: int,
    best_lr: float,
    best_agg: int,
    client_counts: List[int] = CLIENT_COUNTS,
    output_dir: str = "results/scalability_best",
) -> dict:
    """
    Run Independent QHD + FedQHD for each N in client_counts using the best
    hyperparameters found by the search.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[int, dict] = {}

    print("\n" + "=" * 70)
    print(f"Scalability Run  |  lr={best_lr}  agg={best_agg}  eps={episodes}")
    print("=" * 70)

    for N in client_counts:
        print(f"\n--- N = {N} clients ---")
        args = make_args(
            env=env,
            episodes=episodes,
            agent_num=N,
            learning_rate=best_lr,
            aggregation_interval=best_agg,
        )

        ind = train_independent_qhd(episodes, args)
        fed = train_fedqhd_homogeneous(episodes, args)

        ind_final = np.mean(ind.reward_history[-50:]) if len(ind.reward_history) >= 50 else np.mean(ind.reward_history)
        fed_final = np.mean(fed.reward_history[-50:]) if len(fed.reward_history) >= 50 else np.mean(fed.reward_history)
        threshold = max(ind_final * 0.70, 1.0)
        spd = speedup_ratio(fed.reward_history, ind.reward_history, threshold)

        print(f"  Independent: {ind.final_avg_reward:.2f}  FedQHD: {fed.final_avg_reward:.2f}  speedup: {spd:.2f}x")

        all_results[N] = {
            "independent": ind.to_dict(),
            "fedqhd": fed.to_dict(),
            "speedup": spd,
            "threshold": threshold,
        }

        # Save per-N
        n_dir = os.path.join(output_dir, f"N_{N}")
        os.makedirs(n_dir, exist_ok=True)
        with open(os.path.join(n_dir, "independent.json"), "w") as f:
            json.dump(ind.to_dict(), f, indent=2)
        with open(os.path.join(n_dir, "fedqhd.json"), "w") as f:
            json.dump(fed.to_dict(), f, indent=2)

    # Summary
    summary = {
        "config": {"lr": best_lr, "agg_interval": best_agg, "episodes": episodes, "env": env},
        "results": {
            str(N): {
                "independent_final": float(np.mean(all_results[N]["independent"]["reward_history"][-50:])),
                "fedqhd_final": float(np.mean(all_results[N]["fedqhd"]["reward_history"][-50:])),
                "speedup": all_results[N]["speedup"],
            }
            for N in client_counts
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


# ---------------------------------------------------------------------------
# 3. Combined single-plot learning curves
# ---------------------------------------------------------------------------

# Colour cycle: one colour per N, FedQHD solid, Independent dashed
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_combined_learning_curves(
    all_results: dict,
    output_dir: str,
    smooth_window: int = 30,
    title: str = "FedQHD Scalability: Learning Curves",
):
    """
    Single plot with all N values.
      - FedQHD curves: solid lines (one colour per N)
      - Independent baselines: dashed grey lines (one per N, labelled lightly)
      - Vertical markers at convergence threshold crossing
    """
    os.makedirs(output_dir, exist_ok=True)
    client_counts = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Independent baselines (grey dashes, annotated once)
    plotted_ind_label = False
    for i, N in enumerate(client_counts):
        ind_h = all_results[N]["independent"]["reward_history"]
        ind_s = smooth(ind_h, smooth_window)
        label = "Independent QHD" if not plotted_ind_label else "_nolegend_"
        ax.plot(
            ind_s, linestyle="--", color=COLORS[i % len(COLORS)],
            alpha=0.35, linewidth=1.4, label=label,
        )
        plotted_ind_label = True

    # FedQHD curves
    for i, N in enumerate(client_counts):
        fed_h = all_results[N]["fedqhd"]["reward_history"]
        fed_s = smooth(fed_h, smooth_window)
        spd = all_results[N].get("speedup", 0.0)
        spd_str = f"{spd:.1f}×" if np.isfinite(spd) and spd > 0 else "—"
        ax.plot(
            fed_s, linestyle="-", color=COLORS[i % len(COLORS)],
            linewidth=2.2, label=f"FedQHD  N={N}  (speedup {spd_str})",
        )

        # Mark convergence threshold crossing
        thr = all_results[N].get("threshold")
        if thr is not None:
            ep = episodes_to_threshold(fed_h, thr, smooth_window)
            if ep is not None:
                ax.axvline(ep, color=COLORS[i % len(COLORS)], linestyle=":",
                           alpha=0.55, linewidth=1.2)
                ax.annotate(
                    f"N={N}", xy=(ep, fed_s[ep]),
                    xytext=(ep + 3, fed_s[ep] + 1.5),
                    fontsize=7, color=COLORS[i % len(COLORS)],
                )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward (smoothed)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"scalability_combined.{ext}")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()


def plot_speedup_bar(
    all_results: dict,
    output_dir: str,
):
    """Bar chart of convergence speedup per N."""
    os.makedirs(output_dir, exist_ok=True)
    client_counts = sorted(all_results.keys())
    speedups = [
        min(all_results[N].get("speedup", 0.0), 6.0)
        for N in client_counts
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [str(N) for N in client_counts],
        speedups,
        color=[COLORS[i % len(COLORS)] for i in range(len(client_counts))],
        edgecolor="black", linewidth=0.7,
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="No speedup (1×)")
    ax.set_xlabel("Number of Clients (N)", fontsize=12)
    ax.set_ylabel("Convergence Speedup (×)", fontsize=12)
    ax.set_title("FedQHD Convergence Speedup vs. Independent QHD", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, spd in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{spd:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(output_dir, f"scalability_speedup_bar.{ext}")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Load results from saved JSON (for --plot_existing)
# ---------------------------------------------------------------------------

def load_results_from_dir(data_dir: str) -> dict:
    """
    Load per-N result JSON files from a scalability output directory.
    Expects sub-directories N_2/, N_5/, N_10/, N_20/ each containing
    independent.json and fedqhd.json.
    """
    all_results = {}
    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith("N_"):
            continue
        N = int(entry.split("_")[1])
        n_dir = os.path.join(data_dir, entry)
        ind_path = os.path.join(n_dir, "independent.json")
        fed_path = os.path.join(n_dir, "fedqhd.json")
        if not (os.path.exists(ind_path) and os.path.exists(fed_path)):
            continue
        with open(ind_path) as f:
            ind_data = json.load(f)
        with open(fed_path) as f:
            fed_data = json.load(f)

        ind_h = ind_data["reward_history"]
        fed_h = fed_data["reward_history"]
        ind_final = float(np.mean(ind_h[-50:])) if len(ind_h) >= 50 else float(np.mean(ind_h))
        threshold = max(ind_final * 0.70, 1.0)
        spd = speedup_ratio(fed_h, ind_h, threshold)

        all_results[N] = {
            "independent": ind_data,
            "fedqhd": fed_data,
            "speedup": spd,
            "threshold": threshold,
        }
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FedQHD Scalability: hyperparam search + combined learning-curve plot"
    )
    parser.add_argument("--env", default="CartPole",
                        choices=["CartPole", "Acrobot", "LunarLander", "MountainCar", "Taxi"])
    parser.add_argument("--search", action="store_true",
                        help="Run hyperparameter search only (step 1)")
    parser.add_argument("--plot_existing", action="store_true",
                        help="Load existing results and re-plot (skip training)")
    parser.add_argument("--search_episodes", type=int, default=300,
                        help="Episodes for hyperparam search (default 300)")
    parser.add_argument("--search_agents", type=int, default=10,
                        help="Number of agents for search (default 10)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episodes for full scalability run (default 500)")
    parser.add_argument("--client_counts", type=str, default="2,5,10,20",
                        help="Comma-separated client counts (default 2,5,10,20)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override best_lr (skip search)")
    parser.add_argument("--agg", type=int, default=None,
                        help="Override best_agg_interval (skip search)")
    parser.add_argument("--smooth_window", type=int, default=30,
                        help="Smoothing window for plots (default 30)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default results/<env>/scalability_analysis)")
    args = parser.parse_args()

    env = args.env
    client_counts = [int(x.strip()) for x in args.client_counts.split(",")]

    output_dir = args.output_dir or f"results/{env}/scalability_analysis"
    search_dir = os.path.join(output_dir, "hyperparam_search")
    scalability_dir = os.path.join(output_dir, "scalability_best")
    plots_dir = os.path.join(output_dir, "plots")

    # -----------------------------------------------------------------------
    # --plot_existing: load saved data and re-generate plots
    # -----------------------------------------------------------------------
    if args.plot_existing:
        print(f"Loading existing results from {scalability_dir}")
        all_results = load_results_from_dir(scalability_dir)
        if not all_results:
            print(f"No results found in {scalability_dir}. Run without --plot_existing first.")
            return
        plot_combined_learning_curves(all_results, plots_dir,
                                      smooth_window=args.smooth_window)
        plot_speedup_bar(all_results, plots_dir)
        return

    # -----------------------------------------------------------------------
    # Step 1: Hyperparameter search
    # -----------------------------------------------------------------------
    if args.lr is not None and args.agg is not None:
        best_lr, best_agg = args.lr, args.agg
        print(f"Using user-supplied hyperparams: lr={best_lr}  agg_interval={best_agg}")
    else:
        best_lr, best_agg, _ = run_hyperparam_search(
            env=env,
            search_episodes=args.search_episodes,
            search_agents=args.search_agents,
            output_dir=search_dir,
        )

    if args.search:
        print("\nSearch complete. Re-run without --search to run full scalability.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Full scalability run with best params
    # -----------------------------------------------------------------------
    all_results = run_scalability(
        env=env,
        episodes=args.episodes,
        best_lr=best_lr,
        best_agg=best_agg,
        client_counts=client_counts,
        output_dir=scalability_dir,
    )

    # -----------------------------------------------------------------------
    # Step 3: Plots
    # -----------------------------------------------------------------------
    plot_combined_learning_curves(all_results, plots_dir,
                                  smooth_window=args.smooth_window)
    plot_speedup_bar(all_results, plots_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Best lr={best_lr}  agg_interval={best_agg}")
    print(f"  Plots → {plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
