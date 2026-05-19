'''
Visualization code for scalability analysis of FedQHD. Loads results generated
by scalability_experiments.py and re-plots all figures.
'''

import os
import argparse
from typing import Dict
import json
import matplotlib.pyplot as plt
import numpy as np
from experiments_runner import ExperimentResults


def load_scalability_results(results_base_dir) -> Dict[int, Dict[str, ExperimentResults]]:
    """Load scalability results from directory structure: results_base_dir/N_{num_clients}/{method}.json"""
    all_results = {}
    for filename in sorted(os.listdir(results_base_dir)):
        if not os.path.isdir(os.path.join(results_base_dir, filename)):
            continue
        if not filename.startswith("N_"):
            continue
        N = int(filename.split("_")[1])  # Extract N from "N_10" format
        all_results[N] = {}
        for experiment_filename in os.listdir(os.path.join(results_base_dir, filename)):
            if not experiment_filename.endswith(".json"):
                continue
            with open(os.path.join(results_base_dir, filename, experiment_filename), "r") as f:
                data = json.load(f)
            result = ExperimentResults()
            for k, v in data.items():
                setattr(result, k, v)
            method_key = experiment_filename.replace(".json", "")
            all_results[N][method_key] = result

    print(f"Loaded scalability results for {len(all_results)} different client counts: {sorted(all_results.keys())}")
    return all_results


def smooth(values, window_size):
    """Apply moving-average smoothing."""
    if window_size <= 1:
        return np.array(values)
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode='valid')


def plot_scalability_results(
    all_results: Dict[int, Dict[str, ExperimentResults]],
    output_dir: str,
    window_size: int = 20,
):
    """Re-generate all scalability figures from loaded results."""
    os.makedirs(output_dir, exist_ok=True)

    client_counts = sorted(all_results.keys())

    # Collect per-N scalar metrics
    independent_rewards, fedqhd_rewards = [], []
    independent_times, fedqhd_times = [], []
    improvements = []

    for N in client_counts:
        ind = all_results[N]['Independent']
        fed = all_results[N]['FedQHD']
        independent_rewards.append(ind.final_avg_reward)
        fedqhd_rewards.append(fed.final_avg_reward)
        independent_times.append(ind.training_time)
        fedqhd_times.append(fed.training_time)
        improvements.append(fed.final_avg_reward - ind.final_avg_reward)

    # ── Figure 1: Final performance vs N ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(client_counts, independent_rewards, 'o-', label='Independent QHD', linewidth=2)
    ax.plot(client_counts, fedqhd_rewards, 's-', label='FedQHD', linewidth=2)
    ax.set_xlabel('Number of Clients (N)', fontsize=12)
    ax.set_ylabel('Final Average Reward\n(last 100 episodes)', fontsize=12)
    ax.set_title('Scalability: Performance vs. Number of Clients', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'scalability_performance.{ext}'), dpi=300)
    plt.close(fig)

    # ── Figure 2: Improvement vs N ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(client_counts, improvements, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Clients (N)', fontsize=12)
    ax.set_ylabel('Reward Improvement\n(FedQHD − Independent)', fontsize=12)
    ax.set_title('Scalability: FedQHD Improvement over Independent Learning', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'scalability_improvement.{ext}'), dpi=300)
    plt.close(fig)

    # ── Figure 3: Training time vs N ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(client_counts, independent_times, 'o-', label='Independent QHD', linewidth=2)
    ax.plot(client_counts, fedqhd_times, 's-', label='FedQHD', linewidth=2)
    ax.set_xlabel('Number of Clients (N)', fontsize=12)
    ax.set_ylabel('Training Time (s)', fontsize=12)
    ax.set_title('Scalability: Training Time vs. Number of Clients', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'scalability_time.{ext}'), dpi=300)
    plt.close(fig)

    # ── Figure 4: Smoothed learning curves for each N ─────────────────────
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for idx, N in enumerate(client_counts):
        color = colors[idx % len(colors)]
        ind_raw = all_results[N]['Independent'].reward_history
        fed_raw = all_results[N]['FedQHD'].reward_history

        ind_sm = smooth(ind_raw, window_size)
        fed_sm = smooth(fed_raw, window_size)
        ep_ind = range(window_size, len(ind_raw) + 1)
        ep_fed = range(window_size, len(fed_raw) + 1)

        axes[0].plot(ep_ind, ind_sm, color=color, label=f'N={N}', linewidth=1.5)
        axes[1].plot(ep_fed, fed_sm, color=color, label=f'N={N}', linewidth=1.5)

    for ax, title in zip(axes, ['Independent QHD', 'FedQHD']):
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Average Reward', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Scalability: Learning Curves for Different Client Counts', fontsize=13)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'scalability_learning_curves.{ext}'), dpi=300)
    plt.close(fig)

    print(f"Scalability plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Re-plot scalability figures from saved results')
    parser.add_argument('--env', type=str, default='LunarLander',
                        choices=['CartPole', 'Acrobot', 'LunarLander', 'MountainCar'])
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results directory (default: results/scalability/{env})')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save plots (default: same as results_dir)')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Smoothing window for learning curves')
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join('results', 'scalability', args.env)
    output_dir = args.output_dir or results_dir

    all_results = load_scalability_results(results_dir)
    plot_scalability_results(all_results, output_dir, window_size=args.window_size)


if __name__ == "__main__":
    main()
