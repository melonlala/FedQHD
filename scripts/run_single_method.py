#!/usr/bin/env python3
"""
Run a single method for a specific environment and research question.

Replaces only that method's result file in the existing results directory,
merges the entry in summary.json, and writes a params file for traceability.

Usage examples:
    python run_single_method.py --method fedqhd_homo --env CartPole --question q1 --episodes 600
    python run_single_method.py --method fedavg_dqn  --env Acrobot  --question q1 --episodes 600 --dqn_lr 5e-4
    python run_single_method.py --method fedqhd_hetero --env CartPole --question q2 --episodes 400
    python run_single_method.py --method oracle_qhd  --env CartPole --question q1 --episodes 600 --qhd_lr 0.05
    python run_single_method.py --method oracle_dqn_hetero --env CartPole --question q2 --episodes 600

Available methods per question:
  q1: independent_qhd, fedqhd_homo, oracle_qhd, oracle_dqn, fedavg_dqn, distillation_dqn, truncate_fedavg_qhd
  q2: fedqhd_hetero, oracle_qhd_hetero, oracle_dqn_hetero, truncate_fedavg_qhd_hetero, distillation_dqn_hetero
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments_runner import (
    train_independent_qhd,
    train_oracle_qhd,
    train_fedqhd_homogeneous,
    train_fedqhd_heterogeneous,
    train_fedavg_dqn,
    train_oracle_dqn,
    train_truncate_fedavg_qhd,
    train_distillation_dqn,
    ExperimentResults,
)

# ── method registry ────────────────────────────────────────────────────────────
# Maps CLI key → (display name, callable)
# Training functions read args.qhd_lr / args.dqn_lr / etc. directly.
METHODS = {
    # Q1
    'independent_qhd':         ("Independent QHD",
                                 lambda ep, a: train_independent_qhd(ep, a)),
    'fedqhd_homo':             ("FedQHD (Homogeneous)",
                                 lambda ep, a: train_fedqhd_homogeneous(ep, a)),
    'oracle_qhd':              ("Oracle QHD (Homogeneous)",
                                 lambda ep, a: train_oracle_qhd(ep, a, use_heterogeneous=False)),
    'oracle_dqn':              ("Oracle DQN (Homogeneous)",
                                 lambda ep, a: train_oracle_dqn(ep, a)),
    'fedavg_dqn':              ("FedAvg-DQN",
                                 lambda ep, a: train_fedavg_dqn(ep, a)),
    'distillation_dqn':        ("Distillation FedDQN (Homogeneous)",
                                 lambda ep, a: train_distillation_dqn(ep, a, heterogeneous=False)),                        
    'truncate_fedavg_qhd':       ("Truncate FedAvg-QHD (Homogeneous)",
                                   lambda ep, a: train_truncate_fedavg_qhd(ep, a, heterogeneous=False)),
    # Q2
    'independent_qhd_hetero': ("Independent QHD (Heterogeneous)",
                                 lambda ep, a: train_independent_qhd(ep, a, heterogeneous=True)),
    'fedqhd_hetero':           ("FedQHD (Heterogeneous)",
                                 lambda ep, a: train_fedqhd_heterogeneous(ep, a)),
    'oracle_qhd_hetero':       ("Oracle QHD (Heterogeneous)",
                                 lambda ep, a: train_oracle_qhd(ep, a, use_heterogeneous=True)),
    'oracle_dqn_hetero':       ("Oracle DQN (Heterogeneous)",
                                 lambda ep, a: train_oracle_dqn(ep, a, use_heterogeneous=True)),
    'truncate_fedavg_qhd_hetero':("Truncate FedAvg-QHD (Heterogeneous)",
                                   lambda ep, a: train_truncate_fedavg_qhd(ep, a, heterogeneous=True)),
    'distillation_dqn_hetero': ("Distillation FedDQN (Heterogeneous)",
                                 lambda ep, a: train_distillation_dqn(ep, a, heterogeneous=True)),
}

Q1_METHODS = {'independent_qhd', 'fedqhd_homo', 'oracle_qhd',
              'oracle_dqn', 'fedavg_dqn', 'distillation_dqn', 'truncate_fedavg_qhd'}
Q2_METHODS = {'independent_qhd_hetero', 'fedqhd_hetero', 'oracle_qhd_hetero', 'oracle_dqn_hetero',
              'truncate_fedavg_qhd_hetero', 'distillation_dqn_hetero'}


def result_filename(display_name: str) -> str:
    """Convert a method display name to its JSON filename (matches save_results logic)."""
    return display_name.replace(' ', '_') + '.json'


def output_dir_for(args) -> str:
    subdir = 'q1_homogeneous' if args.question == 'q1' else 'q2_heterogeneous'
    return os.path.join(args.output_dir, args.env, subdir)


def save_params(params: dict, output_dir: str, method_key: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'params_{method_key}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved → {path}")


def update_summary(result: ExperimentResults, output_dir: str, args):
    """Merge this method's entry into summary.json (create it if absent)."""
    summary_path = os.path.join(output_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {'configuration': {}, 'methods': {}}

    summary['configuration'].update({
        'episodes':            args.episodes,
        'num_agents':          args.agent_num,
        'qhd_agg_interval':     args.qhd_agg_interval,
        'dqn_agg_interval':     args.dqn_agg_interval,
        'qhd_lr':              args.qhd_lr,
        'dqn_lr':              args.dqn_lr,
        'hyperdimension':      args.hyperdimension,
        'environment':         args.env,
    })

    summary['methods'][result.method_name] = {
        'final_avg_reward':    result.final_avg_reward,
        'final_avg_success':   result.final_avg_success,
        'training_time':       result.training_time,
        'convergence_episode': result.convergence_episode,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary updated  → {summary_path}")


def build_params_dict(method_key: str, display_name: str, args) -> dict:
    """Collect all relevant hyperparameters into a flat dict for the params file."""
    return {
        'method_key':              method_key,
        'method_name':             display_name,
        'question':                args.question,
        'env':                     args.env,
        'episodes':                args.episodes,
        'agent_num':               args.agent_num,
        'runs':                    args.runs,
        'random_seed':             args.random_seed,
        # QHD hyperparams
        'qhd_lr':                  args.qhd_lr,
        'qhd_discount':            args.qhd_discount,
        'qhd_exploration_rate':    args.qhd_exploration_rate,
        'qhd_exploration_decay':   args.qhd_exploration_decay,
        'qhd_exploration_min':     args.qhd_exploration_min,
        'qhd_agg_interval':     args.qhd_agg_interval,
        # DQN hyperparams (None → falls back to base value in training fn)
        'dqn_lr':                  args.dqn_lr,
        'dqn_discount':            args.dqn_discount,
        'dqn_exploration_rate':    args.dqn_exploration_rate,
        'dqn_exploration_decay':   args.dqn_exploration_decay,
        'dqn_exploration_min':     args.dqn_exploration_min,
        'dqn_agg_interval':     args.dqn_agg_interval,
        # Other
        'hyperdimension':          args.hyperdimension,
        'rff_gamma':               args.rff_gamma,
        'anchor_set_size':         args.anchor_set_size,
        'timestamp':               time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def run_single(args):
    method_key = args.method
    display_name, train_fn = METHODS[method_key]

    # Validate question/method combination
    if args.question == 'q1' and method_key not in Q1_METHODS:
        print(f"WARNING: '{method_key}' is a Q2 method; switching --question to q2")
        args.question = 'q2'
    elif args.question == 'q2' and method_key not in Q2_METHODS:
        print(f"WARNING: '{method_key}' is a Q1 method; switching --question to q1")
        args.question = 'q1'

    out_dir = output_dir_for(args)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"METHOD  : {display_name}")
    print(f"ENV     : {args.env}   QUESTION: {args.question.upper()}")
    print(f"EPISODES: {args.episodes}   AGENTS: {args.agent_num}   RUNS: {args.runs}")
    print(f"QHD     : lr={args.qhd_lr}  discount={args.qhd_discount}  "
          f"eps={args.qhd_exploration_rate}  decay={args.qhd_exploration_decay}")
    print(f"DQN     : lr={args.dqn_lr}  agg_interval={args.dqn_agg_interval}  discount={args.dqn_discount}  "
          f"eps={args.dqn_exploration_rate}  decay={args.dqn_exploration_decay}")
    print(f"OUTPUT  : {out_dir}")
    print("=" * 70 + "\n")

    np.random.seed(args.random_seed)

    # ── multi-run averaging ────────────────────────────────────────────────
    if args.runs > 1:
        all_results = []
        for run_id in range(args.runs):
            print(f"\n── Run {run_id + 1}/{args.runs} ──")
            args_run = copy.copy(args)
            args_run.random_seed = args.random_seed + run_id
            np.random.seed(args_run.random_seed)
            all_results.append(train_fn(args.episodes, args_run))

        # Average across runs
        result = ExperimentResults()
        result.method_name = display_name
        max_len = max(len(r.reward_history) for r in all_results)
        pad = lambda lst: lst + [lst[-1]] * (max_len - len(lst))
        result.reward_history  = np.mean([pad(r.reward_history)  for r in all_results], axis=0).tolist()
        result.success_history = np.mean([pad(r.success_history) for r in all_results], axis=0).tolist()
        result.training_time   = float(np.mean([r.training_time for r in all_results]))
        result.final_avg_reward  = float(np.mean([r.final_avg_reward  for r in all_results]))
        result.final_avg_success = float(np.mean([r.final_avg_success for r in all_results]))
        std_reward = float(np.std([r.final_avg_reward for r in all_results]))
        if all_results[0].projection_residuals:
            max_rlen = max(len(r.projection_residuals) for r in all_results)
            padr = lambda lst: lst + [lst[-1]] * (max_rlen - len(lst))
            result.projection_residuals = np.mean(
                [padr(r.projection_residuals) for r in all_results], axis=0).tolist()
        print(f"\nAveraged over {args.runs} runs:")
        print(f"  Final reward: {result.final_avg_reward:.2f} ± {std_reward:.2f}")
    else:
        result = train_fn(args.episodes, args)

    # ── save result JSON (replaces existing file) ──────────────────────────
    result_path = os.path.join(out_dir, result_filename(display_name))
    with open(result_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Result saved     → {result_path}")

    # ── update summary.json ────────────────────────────────────────────────
    update_summary(result, out_dir, args)

    # ── save params file ───────────────────────────────────────────────────
    params = build_params_dict(method_key, display_name, args)
    save_params(params, out_dir, method_key)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Final avg reward : {result.final_avg_reward:.2f}")
    print(f"  Training time    : {result.training_time:.2f}s")
    print(f"  Output dir       : {out_dir}")
    print("=" * 70 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run a single FedQHD method and replace its results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Method / environment / question
    parser.add_argument('--method', default='fedqhd_homo', choices=sorted(METHODS.keys()),
                        help='Which method to run')
    parser.add_argument('--env', default='CartPole',
                        choices=['GridWorld', 'LunarLander', 'CartPole', 'MountainCar',
                                 'CliffWalking', 'Acrobot', 'Taxi', 'Pong', 'Freeway'],
                        help='Environment')
    parser.add_argument('--question', default='q2', choices=['q1', 'q2'],
                        help='Research question (determines output subdirectory)')

    # Training
    parser.add_argument('--episodes',    type=int, default=600)
    parser.add_argument('--agent_num',   type=int, default=5)
    parser.add_argument('--runs',        type=int, default=1,
                        help='Number of independent runs to average')
    parser.add_argument('--random_seed', type=int, default=42)

    # QHD hyperparams (used directly by QHD training functions)
    parser.add_argument('--qhd_lr',               type=float, default=0.1)
    parser.add_argument('--qhd_discount',         type=float, default=0.95)
    parser.add_argument('--qhd_exploration_rate', type=float, default=1.0)
    parser.add_argument('--qhd_exploration_decay',type=float, default=0.95)
    parser.add_argument('--qhd_exploration_min',  type=float, default=0.001)
    parser.add_argument('--qhd_agg_interval', type=int, default=25)

    # DQN hyperparams (None → falls back to base value in training fn)
    parser.add_argument('--dqn_lr',               type=float, default=0.001)
    parser.add_argument('--dqn_discount',         type=float, default=0.99)
    parser.add_argument('--dqn_exploration_rate', type=float, default=1.0)
    parser.add_argument('--dqn_exploration_decay',type=float, default=0.995)
    parser.add_argument('--dqn_exploration_min',  type=float, default=0.001)
    parser.add_argument('--dqn_agg_interval', type=int, default=25)

    # Architecture / federation
    parser.add_argument('--hyperdimension',       type=int,   default=10000)
    parser.add_argument('--rff_gamma',            type=float, default=1.0)
    parser.add_argument('--anchor_set_size',      type=int,   default=200)
    parser.add_argument('--hetero_dims',          type=str,   default=None,
                        help='Comma-separated encoder dims for Q2 heterogeneous methods '
                             '(e.g. "2000,4000,6000,10000"). Defaults to [500,1000,2000,5000].')

    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--grid_size',  type=int, nargs=2, default=[8, 8])

    args = parser.parse_args()
    run_single(args)


if __name__ == '__main__':
    main()
