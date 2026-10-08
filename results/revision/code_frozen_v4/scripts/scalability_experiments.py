"""
Scalability Analysis: Vary number of clients N ∈ {5, 10, 20, 50}
Section 3.5 in experiments.tex
"""

import numpy as np
import time
import os
import json
from typing import List, Dict
import matplotlib.pyplot as plt
from experiments_runner import (
    train_independent_qhd,
    train_fedqhd_homogeneous,
    ExperimentResults
)


def run_scalability_analysis(
    client_counts: List[int],
    episodes: int,
    args
) -> Dict[int, Dict[str, ExperimentResults]]:
    """
    Run scalability experiments by varying number of clients.

    Args:
        client_counts: List of client numbers to test (e.g., [5, 10, 20, 50])
        episodes: Number of training episodes per experiment
        args: Configuration arguments

    Returns:
        Dictionary mapping N (client count) to results for each method
    """
    all_results = {}

    for N in client_counts:
        print(f"\n{'='*80}")
        print(f"SCALABILITY EXPERIMENT: N = {N} clients")
        print(f"{'='*80}")

        # Update args for this experiment
        args.agent_num = N

        results_for_N = {}

        # Run Independent QHD (baseline)
        print(f"\nRunning Independent QHD with N={N}...")
        results_for_N['Independent'] = train_independent_qhd(episodes, args)

        # Run FedQHD (homogeneous)
        print(f"\nRunning FedQHD with N={N}...")
        results_for_N['FedQHD'] = train_fedqhd_homogeneous(episodes, args)

        all_results[N] = results_for_N

        # Print summary for this N
        print(f"\n{'='*60}")
        print(f"Summary for N={N}:")
        print(f"Independent QHD:")
        print(f"  - Final reward: {results_for_N['Independent'].final_avg_reward:.2f}")
        print(f"  - Training time: {results_for_N['Independent'].training_time:.2f}s")
        print(f"FedQHD:")
        print(f"  - Final reward: {results_for_N['FedQHD'].final_avg_reward:.2f}")
        print(f"  - Training time: {results_for_N['FedQHD'].training_time:.2f}s")
        print(f"  - Improvement: {results_for_N['FedQHD'].final_avg_reward - results_for_N['Independent'].final_avg_reward:.2f}")
        print(f"{'='*60}")

    return all_results


def plot_scalability_results(
    all_results: Dict[int, Dict[str, ExperimentResults]],
    output_dir: str,
    window_size: int = 10
):
    """
    Create scalability plots showing performance vs. number of clients.

    Generates:
    1. Performance improvement vs. N
    2. Convergence speed vs. N
    3. Training time vs. N
    """
    os.makedirs(output_dir, exist_ok=True)

    client_counts = sorted(all_results.keys())

    # Extract metrics
    independent_rewards = []
    fedqhd_rewards = []
    independent_times = []
    fedqhd_times = []
    improvements = []
    speedups = []

    for N in client_counts:
        ind_reward = all_results[N]['Independent'].final_avg_reward
        fed_reward = all_results[N]['FedQHD'].final_avg_reward
        ind_time = all_results[N]['Independent'].training_time
        fed_time = all_results[N]['FedQHD'].training_time

        independent_rewards.append(ind_reward)
        fedqhd_rewards.append(fed_reward)
        independent_times.append(ind_time)
        fedqhd_times.append(fed_time)
        improvements.append(fed_reward - ind_reward)

        # Convergence speedup (based on reward improvement)
        if ind_reward != 0:
            speedups.append((fed_reward / ind_reward - 1) * 100)  # Percentage improvement
        else:
            speedups.append(0)

    # Plot 1: Performance vs. N
    plt.figure(figsize=(10, 6))
    plt.plot(client_counts, independent_rewards, 'o-', label='Independent QHD', linewidth=2)
    plt.plot(client_counts, fedqhd_rewards, 's-', label='FedQHD', linewidth=2)
    plt.xlabel('Number of Clients (N)', fontsize=12)
    plt.ylabel('Final Average Reward', fontsize=12)
    plt.title('Scalability: Performance vs. Number of Clients', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_performance.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'scalability_performance.png'), dpi=300)
    plt.close()

    # Plot 2: Performance Improvement vs. N
    plt.figure(figsize=(10, 6))
    plt.plot(client_counts, improvements, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Number of Clients (N)', fontsize=12)
    plt.ylabel('Performance Improvement (FedQHD - Independent)', fontsize=12)
    plt.title('Scalability: FedQHD Improvement over Independent Learning', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_improvement.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'scalability_improvement.png'), dpi=300)
    plt.close()

    # Plot 3: Training Time vs. N
    plt.figure(figsize=(10, 6))
    plt.plot(client_counts, independent_times, 'o-', label='Independent QHD', linewidth=2)
    plt.plot(client_counts, fedqhd_times, 's-', label='FedQHD', linewidth=2)
    plt.xlabel('Number of Clients (N)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Scalability: Training Time vs. Number of Clients', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_time.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'scalability_time.png'), dpi=300)
    plt.close()

    # Plot 4: Learning curves for different N (combined)
    # draw curves in one plot with different colors and labels
    # Smooth the curves using a moving average for better visualization, 

    fig = plt.figure(figsize=(12, 8))
    for N in client_counts:
        ind_rewards = all_results[N]['Independent'].reward_history
        fed_rewards = all_results[N]['FedQHD'].reward_history
        episodes_range = range(1, len(ind_rewards) + 1)

        plt.plot(episodes_range, ind_rewards, label=f'Independent N={N}', linestyle='--')
        plt.plot(episodes_range, fed_rewards, label=f'FedQHD N={N}', linestyle='-')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Scalability: Learning Curves for Different Client Counts', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_learning_curves.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'scalability_learning_curves.png'), dpi=300)
    plt.close()

    print(f"\nScalability plots saved to {output_dir}")


def save_scalability_results(
    all_results: Dict[int, Dict[str, ExperimentResults]],
    output_dir: str,
    args
):
    """Save scalability experiment results to JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # Create summary
    summary = {
        'configuration': {
            'client_counts': sorted(all_results.keys()),
            'episodes': args.episodes,
            'environment': args.env,
            'aggregation_interval': args.qhd_agg_interval,
            'learning_rate': args.qhd_lr,
            'hyperdimension': args.hyperdimension
        },
        'results': {}
    }

    for N in sorted(all_results.keys()):
        summary['results'][f'N={N}'] = {
            'Independent': {
                'final_reward': all_results[N]['Independent'].final_avg_reward,
                'training_time': all_results[N]['Independent'].training_time
            },
            'FedQHD': {
                'final_reward': all_results[N]['FedQHD'].final_avg_reward,
                'training_time': all_results[N]['FedQHD'].training_time
            },
            'improvement': all_results[N]['FedQHD'].final_avg_reward - all_results[N]['Independent'].final_avg_reward,
            'speedup': all_results[N]['FedQHD'].training_time / all_results[N]['Independent'].training_time
        }

    # Save full results for each N
    for N in sorted(all_results.keys()):
        n_dir = os.path.join(output_dir, f'N_{N}')
        os.makedirs(n_dir, exist_ok=True)

        for method_name, results in all_results[N].items():
            method_file = os.path.join(n_dir, f'{method_name}.json')
            with open(method_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)

    # Save summary
    summary_file = os.path.join(output_dir, 'scalability_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nScalability results saved to {output_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    import argparse
    from env.bounds import state_bounds

    parser = argparse.ArgumentParser(description='Run scalability experiments')
    parser.add_argument('--env', type=str, default='LunarLander',
                       choices=['CartPole', 'Acrobot', 'LunarLander', 'MountainCar', 'Pong', 'Freeway'])
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--client_counts', type=str, default='1, 2, 5,10,20,40',
                       help='Comma-separated client counts to test')
    
    parser.add_argument('--qhd_lr', type=float, default=0.2,
                       help='Learning rate for QHD methods')
    parser.add_argument('--qhd_agg_interval', type=int, default=10,
                       help='Aggregation interval for QHD methods (alias for --aggregation_interval)')
    parser.add_argument('--qhd_discount', type=float, default=0.99,
                       help='Discount factor for QHD methods')
    parser.add_argument('--qhd_exploration_rate', type=float, default=1.0,
                       help='Initial exploration rate for QHD methods')
    parser.add_argument('--qhd_exploration_decay', type=float, default=0.99,
                       help='Exploration decay for QHD methods')
    parser.add_argument('--qhd_exploration_min', type=float, default=0.001,
                       help='Minimum exploration rate for QHD methods')
    parser.add_argument('--hyperdimension', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='results/scalability')
    parser.add_argument('--rff_gamma', type=float, default=0.5)

    args = parser.parse_args()

    # Parse client counts
    client_counts = [int(x.strip()) for x in args.client_counts.split(',')]

    print(f"\n{'='*80}")
    print(f"SCALABILITY ANALYSIS")
    print(f"Environment: {args.env}")
    print(f"Client counts: {client_counts}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*80}\n")

    # Run experiments
    all_results = run_scalability_analysis(client_counts, args.episodes, args)

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.env)

    # Save results
    save_scalability_results(all_results, output_dir, args)

    # Plot results
    plot_scalability_results(all_results, output_dir)

    print(f"\n{'='*80}")
    print(f"SCALABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
