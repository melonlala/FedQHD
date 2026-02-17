
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import gymnasium as gym
from train import train_fedavg, train_parallel, train_single_agent
from test import test_agent
from utils import (
    visualize_reward_history,
    visualize_success_history,
    plot_combined_rewards,
    plot_parallel_vs_fedavg_comparison
)




def multi_run_comparison(episodes, args):
    """Run multiple training sessions comparing parallel and fedavg"""
    runs = args.runs
    all_parallel_rewards = []
    all_parallel_success = []
    all_fedavg_rewards = []
    all_fedavg_success = []

    for run in range(runs):
        print(f"\n{'='*60}")
        print(f"Starting run {run + 1}/{runs}")
        print(f"{'='*60}")
        
        parallel_start_time = time.time()
        # Train parallel agents
        print(f"\nTraining PARALLEL agents (run {run + 1})...")
        parallel_global_rewards, parallel_global_success, _, _ = train_parallel(
            episodes, args, args.aggregation_interval
        )
        all_parallel_rewards.append(parallel_global_rewards)
        all_parallel_success.append(parallel_global_success)
        parallel_end_time = time.time()
        parallel_training_time = parallel_end_time - parallel_start_time
        print(f"Run {run + 1} parallel training time: {parallel_end_time - parallel_start_time:.2f} seconds")    
        
        
        # Train FedAvg agents
        print(f"\nTraining FEDAVG agents (run {run + 1})...")
        fedavg_start_time = time.time()
        _, fedavg_global_rewards, fedavg_global_success, _, _ = train_fedavg(
            episodes, args
        )
        all_fedavg_rewards.append(fedavg_global_rewards)
        all_fedavg_success.append(fedavg_global_success)
        fedavg_end_time = time.time()
        fedavg_traning_time = fedavg_end_time - fedavg_start_time
        print(f"Run {run + 1} FedAvg training time: {fedavg_end_time - fedavg_start_time:.2f} seconds")

    # Average results across runs
    avg_parallel_rewards = np.mean(all_parallel_rewards, axis=0)
    avg_parallel_success = np.mean(all_parallel_success, axis=0)
    avg_fedavg_rewards = np.mean(all_fedavg_rewards, axis=0)
    avg_fedavg_success = np.mean(all_fedavg_success, axis=0)

    return (avg_parallel_rewards, avg_parallel_success, 
            avg_fedavg_rewards, avg_fedavg_success, parallel_training_time, fedavg_traning_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a Q-learning agent in GridWorld')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                       help='Size of the gridworld (default: (5, 5) for 5x5)')
    parser.add_argument('--episodes', type=int, default=600, 
                       help='Number of training episodes')
    parser.add_argument('--agent_type', type=str, default='hd_sarsa', 
                       choices=['q_learning', 'dqn', 'qhd' ,'sarsa', 'linear_sarsa', 'hd_sarsa'],
                       help='Type of agent to use')
    parser.add_argument('--aggregation_interval', type=int, default=5,
                       help='FedAvg: aggregate every N steps')
    parser.add_argument('--learning_rate', type=float, default=0.1, 
                       help='Learning rate for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=0.8, 
                       help='Discount factor (gamma)')
    parser.add_argument('--exploration_rate', type=float, default=0.5, 
                       help='Initial exploration rate (epsilon)')
    parser.add_argument('--exploration_decay', type=float, default=0.8, 
                       help='Exploration rate decay per episode')
    parser.add_argument('--exploration_min', type=float, default=0.001, 
                       help='Minimum exploration rate')
    parser.add_argument('--output_dir', type=str, default='test_figs', 
                       help='Directory to save output figures')
    parser.add_argument('--avg_window', type=int, default=5,
                       help='Window size for averaging in plots')
    parser.add_argument('--test', action='store_true',
                       help='Test the trained agent after training')
    parser.add_argument('--convergence_threshold', type=float, default=1,
                       help='Success rate threshold for convergence (default: 0.8)')
    parser.add_argument('--convergence_window', type=int, default=20,
                       help='Window size for convergence calculation (default: 50)')
    parser.add_argument('--agent_range', type=str, default='5',
                       help='Comma-separated agent numbers for multi-run (e.g., "1,3,5,7,10")')
    parser.add_argument('--env', type=str, default='Taxi',
                        choices=['GridWorld', 'LunarLander', 'CartPole', 'MountainCar', 'CliffWalking', 'Acrobot', 'Taxi'],)
    parser.add_argument('--runs', type=int, default=1
                        )
    parser.add_argument('--hyperdimension', type=int, default=10000,
                       help='Hyperdimensional vector dimension for HD agents')
    parser.add_argument('--use_structured_rbf', action='store_true',
                       help='Use structured RBF (64 centers on 8x8 grid) instead of random RBF (1000 centers) for linear_sarsa')
    parser.add_argument('--num_rbf_centers', type=int, default=1000,
                       help='Number of RBF centers for random RBF linear_sarsa (default: 1000)')
    parser.add_argument('--anchor_set_sizes', type=int, default=200)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COMPARISON MODE: Parallel vs FedAvg")
    print(f"{'='*60}\n")
    
    agent_numbers = [int(x.strip()) for x in args.agent_range.split(',')]
    
    for num_agents in agent_numbers:
            
        args.agent_num = num_agents
        
        print(f"\n{'='*60}")
        print(f"Comparing {num_agents} agents:")
        print(f"  Agent Type: {args.agent_type}")
        print(f"  Environment: {args.env}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Runs: {args.runs}")
        print(f"  Aggregation Interval: {args.aggregation_interval}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        parallel_rewards, parallel_success, fedavg_rewards, fedavg_success, parallel_training_time, fedavg_training_time = multi_run_comparison(
            episodes=args.episodes, args=args
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nTotal comparison time: {elapsed_time:.2f} seconds")
        
        # Create output directory with aggregation interval and learning rate
        output_dir = f"{args.output_dir}/{args.env}/{num_agents}_{args.agent_type}_agg{args.aggregation_interval}_lr{args.learning_rate}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot comparison
        plot_parallel_vs_fedavg_comparison(
            parallel_rewards, fedavg_rewards,
            num_agents, args.avg_window, 
            args.agent_type, output_dir
        )
        
        # Save experiment configuration
        config_path = os.path.join(output_dir, 'experiment_info.txt')
        with open(config_path, 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write(f"Training time (seconds): {elapsed_time:.2f}\n")
            f.write(f"Parallel training time (seconds): {parallel_training_time:.2f}\n")
            f.write(f"FedAvg training time (seconds): {fedavg_training_time:.2f}\n")
            f.write(f"\nFinal Results (last 100 episodes average):\n")
            f.write(f"Parallel Avg Reward: {np.mean(parallel_rewards[-100:]):.2f}\n")
            f.write(f"FedAvg Avg Reward: {np.mean(fedavg_rewards[-100:]):.2f}\n")
            f.write(f"Parallel Avg Success: {np.mean(parallel_success[-100:]):.2f}\n")
            f.write(f"FedAvg Avg Success: {np.mean(fedavg_success[-100:]):.2f}\n")
