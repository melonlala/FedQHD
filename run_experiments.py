#!/usr/bin/env python3
"""
Master Experiment Runner for FedQHD Paper
Runs all experiments mentioned in sections/experiments.tex

Usage:
    # Run Q1: Homogeneous encoders comparison
    python run_experiments.py --experiment q1 --env CartPole --episodes 600 --agent_num 5

    # Run Q2: Heterogeneous encoders
    python run_experiments.py --experiment q2 --env CartPole --episodes 600 --agent_num 5

    # Run Q4: Scalability analysis
    python run_experiments.py --experiment scalability --env CartPole --episodes 600

    # Run all experiments for a given environment
    python run_experiments.py --experiment all --env CartPole --episodes 600

    # Multi-run experiments with averaging
    python run_experiments.py --experiment q1 --env CartPole --episodes 600 --runs 3
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import (
    run_full_comparison,
    save_results,
    ExperimentResults
)
from scalability_experiments import (
    run_scalability_analysis,
    save_scalability_results,
    plot_scalability_results
)
from visualize_results import visualize_all_results


def run_q1_experiments(args) -> Dict[str, ExperimentResults]:
    """
    Q1: Performance Comparison with Homogeneous Encoders
    Compare: Independent QHD, FedQHD, Oracle QHD, FedAvg-DQN
    """
    print("\n" + "="*80)
    print("Q1: PERFORMANCE COMPARISON - HOMOGENEOUS ENCODERS")
    print("="*80)

    results = run_full_comparison(args.episodes, args)
    return results


def run_q2_experiments(args) -> Dict[str, ExperimentResults]:
    """
    Q2: Heterogeneous Encoders and Anchor-Based Aggregation
    Compare homogeneous vs. heterogeneous FedQHD
    """
    print("\n" + "="*80)
    print("Q2: HETEROGENEOUS ENCODERS AND ANCHOR-BASED AGGREGATION")
    print("="*80)

    args.test_heterogeneous = True
    results = run_full_comparison(args.episodes, args)
    return results


def run_scalability_experiments(args):
    """
    Q4: Scalability Analysis
    Vary number of clients N ∈ {5, 10, 20, 50}
    """
    print("\n" + "="*80)
    print("Q4: SCALABILITY ANALYSIS")
    print("="*80)

    client_counts = [int(x.strip()) for x in args.client_counts.split(',')]
    results = run_scalability_analysis(client_counts, args.episodes, args)

    # Save and visualize
    output_dir = os.path.join(args.output_dir, args.env, 'scalability')
    save_scalability_results(results, output_dir, args)
    plot_scalability_results(results, output_dir)

    return results


def run_multi_run_experiments(args, experiment_func, num_runs=3):
    """
    Run experiments multiple times and average results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} INDEPENDENT RUNS FOR AVERAGING")
    print(f"{'='*80}")

    all_runs = []

    for run_id in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_id + 1}/{num_runs}")
        print(f"{'='*60}")

        # Update random seed for this run
        args.random_seed = 42 + run_id

        # Run experiment
        run_results = experiment_func(args)
        all_runs.append(run_results)

    # Average results across runs
    print(f"\n{'='*60}")
    print("AVERAGING RESULTS ACROSS RUNS")
    print(f"{'='*60}")

    # Get method names from first run
    method_names = list(all_runs[0].keys())

    averaged_results = {}
    for method_name in method_names:
        # Extract results for this method across all runs
        method_runs = [run[method_name] for run in all_runs]

        # Average reward histories (pad to same length)
        max_len = max(len(r.reward_history) for r in method_runs)
        padded_rewards = []
        padded_success = []

        for r in method_runs:
            rewards = r.reward_history + [r.reward_history[-1]] * (max_len - len(r.reward_history))
            success = r.success_history + [r.success_history[-1]] * (max_len - len(r.success_history))
            padded_rewards.append(rewards)
            padded_success.append(success)

        avg_result = ExperimentResults()
        avg_result.method_name = method_name
        avg_result.reward_history = np.mean(padded_rewards, axis=0).tolist()
        avg_result.success_history = np.mean(padded_success, axis=0).tolist()
        avg_result.training_time = np.mean([r.training_time for r in method_runs])
        avg_result.final_avg_reward = np.mean([r.final_avg_reward for r in method_runs])
        avg_result.final_avg_success = np.mean([r.final_avg_success for r in method_runs])

        # For heterogeneous, average projection residuals
        if method_runs[0].projection_residuals:
            max_res_len = max(len(r.projection_residuals) for r in method_runs)
            padded_residuals = []
            for r in method_runs:
                if r.projection_residuals:
                    residuals = r.projection_residuals + [r.projection_residuals[-1]] * (max_res_len - len(r.projection_residuals))
                    padded_residuals.append(residuals)
            if padded_residuals:
                avg_result.projection_residuals = np.mean(padded_residuals, axis=0).tolist()

        averaged_results[method_name] = avg_result

        print(f"{method_name}:")
        print(f"  Avg final reward: {avg_result.final_avg_reward:.2f} ± {np.std([r.final_avg_reward for r in method_runs]):.2f}")
        print(f"  Avg training time: {avg_result.training_time:.2f}s")

    return averaged_results


def main():
    parser = argparse.ArgumentParser(
        description='Run FedQHD Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment selection
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['q1', 'q2', 'scalability', 'all'],
                       help='Which experiment to run')

    # Environment
    parser.add_argument('--env', type=str, default='CartPole',
                       choices=['GridWorld', 'LunarLander', 'CartPole', 'MountainCar',
                               'CliffWalking', 'Acrobot', 'Taxi', 'Pong', 'Freeway'],
                       help='Environment to test on')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes (max, will stop early if converged)')
    parser.add_argument('--agent_num', type=int, default=5,
                       help='Number of agents/clients')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of independent runs for averaging')

    # Agent parameters
    parser.add_argument('--agent_type', type=str, default='hd_sarsa',
                       choices=['q_learning', 'dqn', 'qhd', 'sarsa', 'linear_sarsa', 'hd_sarsa'])
    parser.add_argument('--hyperdimension', type=int, default=10000,
                       help='Hyperdimensional vector dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--exploration_rate', type=float, default=1.0)
    parser.add_argument('--exploration_decay', type=float, default=0.9995)
    parser.add_argument('--exploration_min', type=float, default=0.01)
    parser.add_argument('--rff_gamma', type=float, default=1.0,
                       help='RFF kernel bandwidth parameter')

    # Federation parameters
    parser.add_argument('--aggregation_interval', type=int, default=50,
                       help='Aggregate every N episodes')

    # Scalability parameters
    parser.add_argument('--client_counts', type=str, default='5,10,20,50',
                       help='Comma-separated client counts for scalability (e.g., "5,10,20,50")')

    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations after experiments')

    # Misc
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                       help='Grid size for GridWorld')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set numpy seed
    np.random.seed(args.random_seed)

    print("\n" + "="*80)
    print("FEDQHD EXPERIMENT RUNNER")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Agents: {args.agent_num}")
    print(f"Runs: {args.runs}")
    print(f"Aggregation Interval: {args.aggregation_interval}")
    print(f"Hyperdimension: {args.hyperdimension}")
    print("="*80 + "\n")

    start_time = time.time()

    # Run experiments
    if args.experiment == 'q1':
        if args.runs > 1:
            results = run_multi_run_experiments(args, run_q1_experiments, args.runs)
        else:
            results = run_q1_experiments(args)

        # Save results
        output_dir = os.path.join(args.output_dir, args.env, 'q1_homogeneous')
        save_results(results, output_dir, args)

        if args.visualize:
            visualize_all_results(output_dir)

    elif args.experiment == 'q2':
        if args.runs > 1:
            results = run_multi_run_experiments(args, run_q2_experiments, args.runs)
        else:
            results = run_q2_experiments(args)

        # Save results
        output_dir = os.path.join(args.output_dir, args.env, 'q2_heterogeneous')
        save_results(results, output_dir, args)

        if args.visualize:
            visualize_all_results(output_dir)

    elif args.experiment == 'scalability':
        results = run_scalability_experiments(args)

    elif args.experiment == 'all':
        # Run Q1
        print("\n" + "="*80)
        print("RUNNING ALL EXPERIMENTS")
        print("="*80)

        results_q1 = run_q1_experiments(args)
        output_q1 = os.path.join(args.output_dir, args.env, 'q1_homogeneous')
        save_results(results_q1, output_q1, args)

        # Run Q2
        results_q2 = run_q2_experiments(args)
        output_q2 = os.path.join(args.output_dir, args.env, 'q2_heterogeneous')
        save_results(results_q2, output_q2, args)

        # Run Scalability
        results_scalability = run_scalability_experiments(args)

        # Visualize all
        if args.visualize:
            visualize_all_results(output_q1)
            visualize_all_results(output_q2)

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
