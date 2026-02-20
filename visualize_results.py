"""
Visualization utilities for FedQHD experiments
Generates publication-quality figures for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List
from experiments_runner import ExperimentResults


def smooth_curve(data: List[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to a curve"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves_comparison(
    results_dict: Dict[str, ExperimentResults],
    output_dir: str,
    title: str = "Learning Curves Comparison",
    avg_window: int = 10
):
    """
    Plot learning curves for Q1: Performance Comparison - Homogeneous Encoders
    Figure format for experiments.tex line 82-83
    """
    plt.figure(figsize=(12, 6))

    # Define colors and styles for each method
    method_styles = {
        'Independent QHD': {'color': 'blue', 'linestyle': '--', 'label': 'Independent QHD'},
        'FedQHD': {'color': 'red', 'linestyle': '-', 'label': 'FedQHD (Ours)', 'linewidth': 2.5},
        'Oracle QHD': {'color': 'green', 'linestyle': '-.', 'label': 'Oracle QHD'},
        'FedAvg-DQN': {'color': 'purple', 'linestyle': ':', 'label': 'FedAvg-DQN'},
        'Oracle DQN': {'color': 'orange', 'linestyle': '--', 'label': 'Oracle DQN'},
    }

    for method_name, results in results_dict.items():
        if method_name not in method_styles:
            continue

        style = method_styles[method_name]
        rewards = results.reward_history
        smoothed = smooth_curve(rewards, window=avg_window)

        episodes = np.arange(len(smoothed))
        plt.plot(episodes, smoothed,
                color=style['color'],
                linestyle=style['linestyle'],
                label=style['label'],
                linewidth=style.get('linewidth', 2.0),
                alpha=0.9)

    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Average Reward', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save in multiple formats
    base_name = 'learning_curves_comparison'
    plt.savefig(os.path.join(output_dir, f'{base_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{base_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Learning curves saved to {output_dir}/{base_name}.pdf")


def plot_heterogeneous_comparison(
    homo_results: ExperimentResults,
    hetero_results: ExperimentResults,
    output_dir: str,
    avg_window: int = 10
):
    """
    Plot comparison between homogeneous and heterogeneous encoders
    Figure for experiments.tex line 104 (Figure: hetero_comparison)
    """
    plt.figure(figsize=(12, 6))

    # Homogeneous
    homo_rewards = smooth_curve(homo_results.reward_history, window=avg_window)
    plt.plot(np.arange(len(homo_rewards)), homo_rewards,
            color='blue', linestyle='-', linewidth=2.5,
            label='FedQHD (Homogeneous)', alpha=0.9)

    # Heterogeneous
    hetero_rewards = smooth_curve(hetero_results.reward_history, window=avg_window)
    plt.plot(np.arange(len(hetero_rewards)), hetero_rewards,
            color='red', linestyle='--', linewidth=2.5,
            label='FedQHD (Heterogeneous + Anchor)', alpha=0.9)

    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Average Reward', fontsize=13)
    plt.title('FedQHD: Homogeneous vs. Heterogeneous Encoders', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'hetero_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'hetero_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heterogeneous comparison saved to {output_dir}/hetero_comparison.pdf")


def plot_projection_residual_analysis(
    hetero_results: ExperimentResults,
    output_dir: str
):
    """
    Plot projection residual over communication rounds
    Figure for experiments.tex line 129 (Figure: residual_correlation)
    """
    if not hetero_results.projection_residuals:
        print("No projection residuals to plot")
        return

    residuals = hetero_results.projection_residuals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Residual over rounds
    rounds = np.arange(len(residuals))
    ax1.plot(rounds, residuals, 'o-', color='darkblue', markersize=4, alpha=0.7)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Projection Residual ||R_{i,0}||_F', fontsize=12)
    ax1.set_title('Projection Residual Evolution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of residuals
    ax2.hist(residuals, bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Projection Residual ||R_{i,0}||_F', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'projection_residual_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'projection_residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Projection residual analysis saved to {output_dir}/projection_residual_analysis.pdf")


def create_computation_cost_table(
    results_dict: Dict[str, ExperimentResults],
    output_dir: str
):
    """
    Create LaTeX table for computation cost comparison
    Table for experiments.tex line 115-117 (Computation Cost section)
    """
    table_lines = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\caption{Training Time Comparison (seconds)}")
    table_lines.append("\\label{tab:computation_cost}")
    table_lines.append("\\begin{tabular}{lcc}")
    table_lines.append("\\toprule")
    table_lines.append("Method & Training Time (s) & Relative Time \\\\")
    table_lines.append("\\midrule")

    # Get baseline (Independent) time
    baseline_time = results_dict.get('Independent QHD', None)
    if baseline_time:
        baseline = baseline_time.training_time
    else:
        baseline = 1.0

    for method_name, results in sorted(results_dict.items()):
        time_s = results.training_time
        relative = time_s / baseline if baseline > 0 else 1.0
        table_lines.append(f"{method_name} & {time_s:.2f} & {relative:.2f}x \\\\")

    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")

    # Save to file
    table_file = os.path.join(output_dir, 'computation_cost_table.tex')
    with open(table_file, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"Computation cost table saved to {table_file}")

    # Also save as markdown for easy viewing
    md_lines = ["# Training Time Comparison\n"]
    md_lines.append("| Method | Training Time (s) | Relative Time |")
    md_lines.append("|--------|-------------------|---------------|")

    for method_name, results in sorted(results_dict.items()):
        time_s = results.training_time
        relative = time_s / baseline if baseline > 0 else 1.0
        md_lines.append(f"| {method_name} | {time_s:.2f} | {relative:.2f}x |")

    md_file = os.path.join(output_dir, 'computation_cost_table.md')
    with open(md_file, 'w') as f:
        f.write('\n'.join(md_lines))


def create_performance_summary_table(
    results_dict: Dict[str, ExperimentResults],
    output_dir: str
):
    """
    Create LaTeX table for final performance summary
    """
    table_lines = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\caption{Final Performance Comparison (Last 100 Episodes)}")
    table_lines.append("\\label{tab:performance_summary}")
    table_lines.append("\\begin{tabular}{lccc}")
    table_lines.append("\\toprule")
    table_lines.append("Method & Avg Reward & Avg Success Rate & Training Time (s) \\\\")
    table_lines.append("\\midrule")

    for method_name, results in sorted(results_dict.items()):
        reward = results.final_avg_reward
        success = results.final_avg_success
        time_s = results.training_time
        table_lines.append(f"{method_name} & {reward:.2f} & {success:.3f} & {time_s:.2f} \\\\")

    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")

    # Save to file
    table_file = os.path.join(output_dir, 'performance_summary_table.tex')
    with open(table_file, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"Performance summary table saved to {table_file}")


def visualize_all_results(results_dir: str):
    """
    Load results from JSON files and generate all visualizations
    """
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return

    # Load results
    results_dict = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename != 'summary.json' and not filename.startswith('params_'):
            method_name = filename.replace('.json', '').replace('_', ' ')
            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct ExperimentResults object
            results = ExperimentResults()
            results.method_name = data['method_name']
            results.reward_history = data['reward_history']
            results.success_history = data['success_history']
            results.training_time = data['training_time']
            results.projection_residuals = data.get('projection_residuals', [])
            results.final_avg_reward = data['final_avg_reward']
            results.final_avg_success = data['final_avg_success']

            results_dict[method_name] = results

    if not results_dict:
        print(f"No result files found in {results_dir}")
        return

    print(f"Loaded {len(results_dict)} result files")

    # Generate all visualizations
    plot_learning_curves_comparison(results_dict, results_dir)

    # Check for heterogeneous results
    if 'FedQHD (Heterogeneous)' in results_dict and 'FedQHD (Homogeneous)' in results_dict:
        plot_heterogeneous_comparison(
            results_dict['FedQHD (Homogeneous)'],
            results_dict['FedQHD (Heterogeneous)'],
            results_dir
        )

        plot_projection_residual_analysis(
            results_dict['FedQHD (Heterogeneous)'],
            results_dir
        )

    create_computation_cost_table(results_dict, results_dir)
    create_performance_summary_table(results_dict, results_dir)

    print(f"\nAll visualizations saved to {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize FedQHD experiment results')
    parser.add_argument('--results_dir', type=str, default='results/MountainCar/q1_homogeneous',
                       help='Directory containing experiment JSON results')

    args = parser.parse_args()

    visualize_all_results(args.results_dir)
