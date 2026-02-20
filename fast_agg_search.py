#!/usr/bin/env python3
"""
Fast aggregation interval search for CartPole QHD and Acrobot DQN.
Uses 200 episodes, 1 run, tests agg ∈ {5, 10, 25, 50, 100}.
"""

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import train_fedqhd_homogeneous, train_fedavg_dqn, ExperimentResults


class SimpleArgs:
    """Minimal args namespace for experiment functions."""
    def __init__(self, **kwargs):
        # Defaults
        self.agent_num = 5
        self.hyperdimension = 10000
        self.anchor_set_size = 200
        self.exploration_min = 0.01
        self.rff_gamma = 1.0
        self.random_seed = 42
        self.grid_size = [8, 8]
        self.test_heterogeneous = False
        for k, v in kwargs.items():
            setattr(self, k, v)


def search_cartpole_qhd():
    """Search aggregation interval for CartPole FedQHD Homogeneous."""
    print("\n" + "="*70)
    print("SEARCH: CartPole / FedQHD Homogeneous aggregation interval")
    print("Fixed: lr=0.2, discount=0.9, eps=0.8, eps_decay=0.9")
    print("="*70)

    agg_candidates = [5, 10, 25, 50, 100]
    results = {}

    for agg in agg_candidates:
        print(f"\n--- agg={agg} ---")
        args = SimpleArgs(
            env='CartPole',
            learning_rate=0.2,
            aggregation_interval=agg,
            discount_factor=0.9,
            exploration_rate=0.8,
            exploration_decay=0.9,
        )
        np.random.seed(42)
        r = train_fedqhd_homogeneous(200, args)
        final_reward = np.mean(r.reward_history[-50:])
        results[agg] = {'final_reward': final_reward, 'training_time': r.training_time}
        print(f"  agg={agg:3d} → mean last-50 reward = {final_reward:.2f}  ({r.training_time:.1f}s)")

    best_agg = max(results, key=lambda a: results[a]['final_reward'])
    print(f"\n>>> Best CartPole QHD agg = {best_agg}  "
          f"(reward {results[best_agg]['final_reward']:.2f})")
    print("\nFull results:")
    for agg in agg_candidates:
        marker = " <-- BEST" if agg == best_agg else ""
        print(f"  agg={agg:3d}: {results[agg]['final_reward']:.2f}{marker}")
    return best_agg, results


def search_acrobot_dqn():
    """Search aggregation interval for Acrobot FedAvg-DQN."""
    print("\n" + "="*70)
    print("SEARCH: Acrobot / FedAvg-DQN aggregation interval")
    print("Fixed: lr=0.001, discount=0.99, eps=1.0, eps_decay=0.995")
    print("="*70)

    agg_candidates = [5, 10, 25, 50, 100]
    results = {}

    for agg in agg_candidates:
        print(f"\n--- agg={agg} ---")
        args = SimpleArgs(
            env='Acrobot',
            learning_rate=0.001,
            aggregation_interval=agg,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
        )
        np.random.seed(42)
        r = train_fedavg_dqn(200, args)
        final_reward = np.mean(r.reward_history[-50:])
        results[agg] = {'final_reward': final_reward, 'training_time': r.training_time}
        print(f"  agg={agg:3d} → mean last-50 reward = {final_reward:.2f}  ({r.training_time:.1f}s)")

    best_agg = max(results, key=lambda a: results[a]['final_reward'])
    print(f"\n>>> Best Acrobot DQN agg = {best_agg}  "
          f"(reward {results[best_agg]['final_reward']:.2f})")
    print("\nFull results:")
    for agg in agg_candidates:
        marker = " <-- BEST" if agg == best_agg else ""
        print(f"  agg={agg:3d}: {results[agg]['final_reward']:.2f}{marker}")
    return best_agg, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['cartpole_qhd', 'acrobot_dqn', 'both'], default='both')
    args = parser.parse_args()

    if args.target in ('cartpole_qhd', 'both'):
        best_cp, cp_results = search_cartpole_qhd()

    if args.target in ('acrobot_dqn', 'both'):
        best_ac, ac_results = search_acrobot_dqn()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if args.target in ('cartpole_qhd', 'both'):
        print(f"CartPole QHD best agg = {best_cp}")
    if args.target in ('acrobot_dqn', 'both'):
        print(f"Acrobot  DQN best agg = {best_ac}")
