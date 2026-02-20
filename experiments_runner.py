"""
Comprehensive Experiment Runner for FedQHD Paper
Supports all baselines and experiment types mentioned in sections/experiments.tex
"""

import copy
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple
import tqdm
import torch
import torch.nn.functional as F
from env.utils import create_env
from env.bounds import state_bounds
from agent.qhd_agent import QHDAgent
from agent.dqn_agent import DQNAgent
from agent.fedavg import FedAvgAgent



class ExperimentResults:
    """Container for experiment results"""
    def __init__(self):
        self.method_name = ""
        self.reward_history = []
        self.success_history = []
        self.training_time = 0.0
        self.convergence_episode = None
        self.projection_residuals = []  # For heterogeneous case
        self.final_avg_reward = 0.0
        self.final_avg_success = 0.0

    def to_dict(self):
        return {
            'method_name': self.method_name,
            'reward_history': self.reward_history,
            'success_history': self.success_history,
            'training_time': self.training_time,
            'convergence_episode': self.convergence_episode,
            'projection_residuals': self.projection_residuals,
            'final_avg_reward': self.final_avg_reward,
            'final_avg_success': self.final_avg_success
        }


def train_independent_qhd(episodes: int, args) -> ExperimentResults:
    """
    Baseline 1: Independent QHD
    A randomly selected local agent during each aggregation learning independently without federation.
    This serves as the lower bound.
    """
    results = ExperimentResults()
    results.method_name = "Independent QHD"

    print(f"\n{'='*60}")
    print("Training Independent QHD (no federation)")
    print(f"{'='*60}")

    start_time = time.time()

    # Create fedavg environment and agent
    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    fed_agent = FedAvgAgent(
        state_dim=envs[0].state_dim,
        action_dim=envs[0].action_dim,
        agent_type='qhd',
        num_agents=num_agents,
        learning_rate=args.qhd_lr,
        discount_factor=args.qhd_discount,
        exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
        exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
        exploration_min=getattr(args, 'qhd_exploration_min', 0.001),
        hd_dim=args.hyperdimension,
        state_bounds=state_bounds.get(args.env)
    )
    reward_history = []
    success_history = []
    for episode in tqdm.tqdm(range(episodes), desc="Independent QHD"):
        episode_rewards = []
        episode_successes = []

        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)  # Get the local agent
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)
        
        fed_agent.decay_exploration()  # Decay exploration for all agents
        
        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


        if (episode + 1) % args.qhd_agg_interval == 0:
            
            fed_agent.aggregate()
            #instead pf distributing the global model, we keep the local models randomly initialized every aggregation round to simulate the independent learning scenario
            fed_agent = FedAvgAgent(
                state_dim=envs[0].state_dim,
                action_dim=envs[0].action_dim,
                agent_type='qhd',
                num_agents=num_agents,
                learning_rate=args.qhd_lr,
                discount_factor=args.qhd_discount,
                exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
                exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
                exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
                hd_dim=args.hyperdimension,
                state_bounds=state_bounds.get(args.env)
            )


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"Independent QHD completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_oracle_qhd(episodes: int, args, use_heterogeneous: bool = False) -> ExperimentResults:
    """
    Baseline 2: Oracle QHD (Heterogeneous-Aware)

    Server collects ALL raw data from all clients and trains N separate Q-functions.
    Each Q_i uses client i's encoder Φ_i but is trained on ALL pooled data.

    This provides the true upper bound for heterogeneous case:
    - Perfect data sharing (all clients contribute all data)
    - Each client gets parameters optimized on global data
    - Parameters remain compatible with each client's local encoder

    Args:
        episodes: Number of training episodes
        args: Configuration arguments
        use_heterogeneous: If True, use heterogeneous encoders; else use homogeneous
    """
    results = ExperimentResults()
    results.method_name = "Oracle QHD (Heterogeneous)" if use_heterogeneous else "Oracle QHD"

    print(f"\n{'='*60}")
    print(f"Training Oracle QHD {'(heterogeneous-aware)' if use_heterogeneous else '(homogeneous)'}")
    print("Server: Trains N separate Q-functions on pooled data")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Create N agents with heterogeneous encoders (if requested)
    agents = []
    if use_heterogeneous:
        # Heterogeneous dimensions (same as FedQHD heterogeneous)
        hd_dims = [1000, 5000, 10000, 50000]
        agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]

        for i in range(num_agents):
            # Each client has different bandwidth and dimension
            sigma_i = args.rff_gamma * np.random.uniform(0.5, 1.5)
            agent = QHDAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hd_dim=agent_dims[i],
                rff_gamma=sigma_i,
                learning_rate=args.qhd_lr,
                discount_factor=args.qhd_discount,
                exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
                exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
                exploration_min=args.qhd_exploration_min,
                state_bounds=state_bounds.get(args.env),
                random_seed=42 + i
            )
            agents.append(agent)
        print(f"Using heterogeneous encoders: {agent_dims}")
    else:
        # Homogeneous: all agents share same encoder configuration
        for i in range(num_agents):
            agent = QHDAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hd_dim=args.hyperdimension,
                learning_rate=args.qhd_lr,
                discount_factor=args.qhd_discount,
                exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
                exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
                exploration_min=args.qhd_exploration_min,
                state_bounds=state_bounds.get(args.env),
                random_seed=42 + i
            )
            agents.append(agent)

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc=f"Oracle QHD {'(Hetero)' if use_heterogeneous else '(Homo)'}"):
        episode_rewards = []
        episode_successes = []

        # For each client environment, collect the raw data and aggregation them during global server training
        for env_id, env in enumerate(envs):
            state, _ = env.reset()
            done = False

            # Track rewards for this client's environment
            client_rewards = []

            while not done:
                # Each agent makes its own decision, but we use agent env_id for consistency
                # (In practice, Oracle has access to all policies, so we can use any agent)
                # We'll use the corresponding agent for action selection
                action = agents[env_id].choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # KEY: Train ALL N agents on this transition (data pooling)
                for agent in agents:
                    agent.update_model(state, action, reward, next_state, done)

                state = next_state
                client_rewards.append(reward)

            # Record performance using the agent corresponding to this environment
            total_reward = sum(client_rewards)
            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # Decay exploration for all agents
        for agent in agents:
            agent.decay_exploration()

        # Average across all client environments
        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"Oracle QHD completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")
    if use_heterogeneous:
        print(f"Note: Each client has encoder-compatible parameters trained on global data")

    return results


def train_fedqhd_homogeneous(episodes: int, args) -> ExperimentResults:
    """
    Main Method: FedQHD with Homogeneous Encoders
    All clients use the same encoder (shared RFF parameters).
    Uses direct parameter averaging (Algorithm 1 in methodology.tex).
    """
    results = ExperimentResults()
    results.method_name = "FedQHD (Homogeneous)"

    print(f"\n{'='*60}")
    print("Training FedQHD with Homogeneous Encoders")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]

    # Create FedAvg coordinator with shared encoder
    fed_agent = FedAvgAgent(
        state_dim=envs[0].state_dim,
        action_dim=envs[0].action_dim,
        agent_type='qhd',
        num_agents=num_agents,
        learning_rate=args.qhd_lr,
        discount_factor=args.qhd_discount,
        exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
        exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
        exploration_min=args.qhd_exploration_min,
        hd_dim=args.hyperdimension,
        state_bounds=state_bounds.get(args.env)
    )

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc="FedQHD (Homo)"):
        episode_rewards = []
        episode_successes = []

        # Local training on each client
        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # Aggregate every K episodes (Eq. 148 in methodology.tex)
        if (episode + 1) % args.aggregation_interval == 0:
            fed_agent.aggregate()  # W^glob = sum(pi_i * W_i)
            fed_agent.distribute()

        fed_agent.decay_exploration()

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"FedQHD (Homogeneous) completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_fedqhd_heterogeneous(episodes: int, args) -> ExperimentResults:
    """
    Main Method: FedQHD with Heterogeneous Encoders
    Clients use different encoders (different dimensions, bandwidths).
    Uses anchor-based aggregation (Algorithm 2 in methodology.tex, Eq. 212).
    """
    results = ExperimentResults()
    results.method_name = "FedQHD (Heterogeneous)"

    print(f"\n{'='*60}")
    print("Training FedQHD with Heterogeneous Encoders")
    print(f"Anchor set size: {args.anchor_set_size}")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Heterogeneous dimensions for each client (from experiments.tex line 33)
    hd_dims = [1000, 5000, 10000, 50000]
    agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]

    # Create agents with heterogeneous encoders
    agents = []
    for i in range(num_agents):
        # Each client has different bandwidth and dimension
        sigma_i = args.rff_gamma * np.random.uniform(0.5, 1.5)  # Heterogeneous bandwidth
        agent = QHDAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hd_dim=agent_dims[i],
            rff_gamma=sigma_i,
            learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
            exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
            exploration_min=args.qhd_exploration_min,
            state_bounds=state_bounds.get(args.env),
            random_seed=42 + i  # Different seed per agent
        )
        agents.append(agent)

    # Server: Construct anchor set (methodology.tex line 36)
    # Collect m=args.anchor_set_size states from random rollouts
    anchor_states = []
    temp_env = create_env(args)
    while len(anchor_states) < args.anchor_set_size:
        state, _ = temp_env.reset()
        anchor_states.append(state)
        for _ in range(50):  # Random rollout steps
            action = np.random.randint(action_dim)
            next_state, _, done, _, _ = temp_env.step(action)
            anchor_states.append(next_state)
            if done or len(anchor_states) >= args.anchor_set_size:
                break
    anchor_states = np.array(anchor_states[:args.anchor_set_size])

    

    reward_history = []
    success_history = []
    projection_residuals = []

    for episode in tqdm.tqdm(range(episodes), desc="FedQHD (Hetero)"):
        episode_rewards = []
        episode_successes = []

        # Local training on each client
        for agent_id in range(num_agents):
            agent = agents[agent_id]
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # Anchor-based aggregation (Eq. 212 in methodology.tex)
        if (episode + 1) % args.aggregation_interval == 0:
            # Step 1: Each client evaluates Q-values on anchor set
            anchor_q_values = []  # Q^ref_i for each client
            anchor_features = []  # X_i for each client

            for agent_id in range(num_agents):
                agent = agents[agent_id]
                # Encode anchor states with client i's encoder (Eq. 182-188)
                X_i = np.array([agent.encode_state(s) for s in anchor_states])  # Shape: (m, D_i)
                anchor_features.append(X_i)

                # Compute Q^ref_i = X_i * W_i (Eq. 188)
                Q_ref_i = np.dot(X_i, agent.model_vectors.T)  # Shape: (m, |A|)
                anchor_q_values.append(Q_ref_i)

            # Step 2: Server averages Q-values (function-space consensus)
            Q_glob_ref = np.mean(anchor_q_values, axis=0)  # Shape: (m, |A|)

            # Step 3: Compile back to each client's parameter space (Eq. 212)
            for agent_id in range(num_agents):
                agent = agents[agent_id]
                X_i = anchor_features[agent_id]

                # Ridge regression solution: W^glob_i = (X_i^H X_i + λI)^-1 X_i^H Q^glob_ref
                lambda_reg = 1e-4  # Regularization parameter
                XtX = X_i.T @ X_i + lambda_reg * np.eye(agent.hd_dim)
                XtQ = X_i.T @ Q_glob_ref
                W_glob_i = np.linalg.solve(XtX, XtQ)  # Shape: (D_i, |A|)

                # Update client's model
                agent.model_vectors = W_glob_i.T  # Shape: (|A|, D_i)

                # Compute projection residual (Proposition 1, Eq. 274)
                Q_reconstructed = X_i @ W_glob_i  # X_i * W^glob_i
                residual = np.linalg.norm(Q_reconstructed - Q_glob_ref, 'fro')
                projection_residuals.append(residual)

        # Decay exploration
        for agent in agents:
            agent.decay_exploration()

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.projection_residuals = projection_residuals
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"FedQHD (Heterogeneous) completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")
    print(f"Avg projection residual: {np.mean(projection_residuals):.4f}")

    return results


def train_fedavg_dqn(episodes: int, args) -> ExperimentResults:
    """
    Baseline 3: FedAvg-DQN
    Federated deep Q-learning with parameter averaging.
    Uses a 2-layer MLP with 128 hidden units per layer.
    """
    results = ExperimentResults()
    results.method_name = "FedAvg-DQN"

    print(f"\n{'='*60}")
    print("Training FedAvg-DQN")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]

    fed_agent = FedAvgAgent(
        state_dim=envs[0].state_dim,
        action_dim=envs[0].action_dim,
        agent_type='dqn',
        num_agents=num_agents,
        learning_rate=args.dqn_lr if args.dqn_lr is not None else args.learning_rate,
        discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
        exploration_rate=getattr(args, 'dqn_exploration_rate', 1.0),
        exploration_decay=getattr(args, 'dqn_exploration_decay', 0.995),
        exploration_min=args.dqn_exploration_min if args.dqn_exploration_min is not None else args.exploration_min,
        hd_dim=None,
        state_bounds=state_bounds.get(args.env)
    )

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc="FedAvg-DQN"):
        episode_rewards = []
        episode_successes = []

        for agent_id in range(num_agents):
            agent = fed_agent.get_agent(agent_id)
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        if (episode + 1) % args.aggregation_interval == 0:
            fed_agent.aggregate()
            fed_agent.distribute()

        fed_agent.decay_exploration()

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"FedAvg-DQN completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_oracle_dqn(episodes: int, args) -> ExperimentResults:
    """
    Baseline: Oracle DQN (Centralized DQN with Pooled Data)
    Upper bound baseline using centralized DQN training with data from all clients.
    Reference: experiments.tex line 48
    """
    results = ExperimentResults()
    results.method_name = "Oracle DQN"

    print(f"\n{'='*60}")
    print("Training Oracle DQN (centralized data pooling)")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]

    # Single centralized DQN agent
    agent = DQNAgent(
        state_dim=envs[0].state_dim,
        action_dim=envs[0].action_dim,
        learning_rate=args.dqn_lr if args.dqn_lr is not None else args.learning_rate,
        discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
        exploration_rate=getattr(args, 'dqn_exploration_rate', 1.0),
        exploration_decay=getattr(args, 'dqn_exploration_decay', 0.995),
        exploration_min=args.dqn_exploration_min if args.dqn_exploration_min is not None else args.exploration_min
    )

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc="Oracle DQN"):
        episode_rewards = []
        episode_successes = []

        # Collect data from all clients (simulating centralized data access)
        for agent_id in range(num_agents):
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        agent.decay_exploration()

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"Oracle DQN completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_truncate_fedavg_qhd(episodes: int, args) -> ExperimentResults:
    """
    Baseline: Truncate/Pad FedAvg-QHD (Naive Heterogeneous Baseline)
    Naive approach: pad shorter vectors or truncate longer vectors to a fixed dimension.
    Reference: experiments.tex line 50
    """
    results = ExperimentResults()
    results.method_name = "Truncate FedAvg-QHD"

    print(f"\n{'='*60}")
    print("Training Truncate/Pad FedAvg-QHD (naive heterogeneous)")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Heterogeneous dimensions for each client
    hd_dims = [1000, 5000, 10000, 50000]
    agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]
    target_dim = args.hyperdimension  # Common dimension for averaging

    # Create agents with heterogeneous encoders
    agents = []
    for i in range(num_agents):
        sigma_i = args.rff_gamma * np.random.uniform(0.5, 1.5)
        agent = QHDAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hd_dim=agent_dims[i],
            learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=getattr(args, 'qhd_exploration_rate', 1.0),
            exploration_decay=getattr(args, 'qhd_exploration_decay', 0.995),
            exploration_min=args.qhd_exploration_min,
            state_bounds=state_bounds.get(args.env),
            rff_gamma=sigma_i
        )
        agents.append(agent)

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc="Truncate FedAvg-QHD"):
        episode_rewards = []
        episode_successes = []

        # Local training
        for agent_id in range(num_agents):
            agent = agents[agent_id]
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # Aggregate every K episodes with padding/truncation
        if (episode + 1) % args.aggregation_interval == 0:
            # Collect all weight matrices
            all_weights = []
            for agent in agents:
                W = agent.model_vectors  # Shape: (action_dim, hd_dim_i)
                # Pad or truncate to target_dim
                if W.shape[1] < target_dim:
                    # Pad with zeros
                    W_padded = np.zeros((action_dim, target_dim))
                    W_padded[:, :W.shape[1]] = W
                    all_weights.append(W_padded)
                else:
                    # Truncate
                    W_truncated = W[:, :target_dim]
                    all_weights.append(W_truncated)

            # Average
            W_global = np.mean(all_weights, axis=0)

            # Distribute back (truncate or pad back to original dimensions)
            for agent_id, agent in enumerate(agents):
                orig_dim = agent_dims[agent_id]
                if orig_dim < target_dim:
                    agent.model_vectors = W_global[:, :orig_dim]
                else:
                    W_expanded = np.zeros((action_dim, orig_dim))
                    W_expanded[:, :target_dim] = W_global
                    agent.model_vectors = W_expanded

        # Decay exploration
        for agent in agents:
            agent.epsilon = max(
                agent.epsilon_min,
                agent.epsilon * agent.epsilon_decay
            )

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"Truncate FedAvg-QHD completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_distillation_dqn(episodes: int, args,
                           heterogeneous: bool = True) -> ExperimentResults:
    """
    Baseline: Federated DQN with Knowledge Distillation (homogeneous or heterogeneous)

    Implements the distillation-based federation approach based on:
      - Policy Distillation (Rusu et al. 2016): KL divergence against soft Q-value targets
      - Distral (Teh et al. 2017): shared distilled policy as softmax of mean Q-values

    Algorithm (per aggregation round):
      1. Each client i trains its DQN locally for K episodes (standard DQN).
      2. Each client evaluates Q-values on a shared distillation set S_d.
      3. Server computes a teacher soft-policy via softmax averaging in Q-space
         (Distral-style):
           Q_teacher(s) = (1/N) Σ_i Q_i(s)
           π_teacher(a|s) = softmax(Q_teacher(s) / τ)
      4. Each client distills the teacher back by minimising KL divergence
         (Policy Distillation, Rusu et al. 2016):
           L_KL(θ_i) = Σ_{s∈S_d} KL( π_teacher(·|s) ‖ softmax(Q_i(s)/τ) )

    Args:
        heterogeneous: If True, each client uses a different hidden-layer width
                       [64, 128, 256, 512], making parameter averaging undefined
                       (Q2 / hetero encoder setting).
                       If False, all clients share hidden_size=128, so architectures
                       are identical (Q1 / homo encoder setting).
    """
    label = "Distillation FedDQN (Hetero)" if heterogeneous else "Distillation FedDQN"
    results = ExperimentResults()
    results.method_name = label

    print(f"\n{'='*60}")
    print(f"Training {label}")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Homogeneous: all clients share the same hidden width (128), identical to
    # standard FedAvg-DQN in architecture; federation still occurs via distillation.
    # Heterogeneous: clients cycle through [64, 128, 256, 512], so parameter
    # averaging is algebraically undefined — only function-space distillation applies.
    if heterogeneous:
        hidden_sizes = [64, 128, 256, 512]
        agent_hidden = [hidden_sizes[i % len(hidden_sizes)] for i in range(num_agents)]
    else:
        agent_hidden = [128] * num_agents
    print(f"  Client hidden sizes: {agent_hidden}")

    # Instantiate one DQN per client with its own architecture
    agents = []
    for i in range(num_agents):
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=agent_hidden[i],
            learning_rate=args.dqn_lr if args.dqn_lr is not None else args.learning_rate,
            discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
            exploration_rate=getattr(args, 'dqn_exploration_rate', 1.0),
            exploration_decay=getattr(args, 'dqn_exploration_decay', 0.995),
            exploration_min=args.dqn_exploration_min if args.dqn_exploration_min is not None else args.exploration_min,
        )
        agents.append(agent)

    # Build distillation set S_d: random rollout states from the environment
    distill_states: List[np.ndarray] = []
    temp_env = create_env(args)
    while len(distill_states) < args.anchor_set_size:
        state, _ = temp_env.reset()
        distill_states.append(state)
        for _ in range(50):
            action = np.random.randint(action_dim)
            next_state, _, done, _, _ = temp_env.step(action)
            distill_states.append(next_state)
            if done or len(distill_states) >= args.anchor_set_size:
                break
    distill_states = distill_states[:args.anchor_set_size]

    # Distillation temperature τ: higher → softer targets (less confident teacher)
    tau = 1.0
    # Number of gradient steps per distillation round
    distill_steps = 10

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc=label):
        episode_rewards = []
        episode_successes = []

        # ── Phase 1: Local DQN training ──────────────────────────────────────
        for agent_id in range(num_agents):
            agent = agents[agent_id]
            env = envs[agent_id]
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update_model(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # ── Phase 2: Distillation aggregation every K episodes ───────────────
        if (episode + 1) % args.aggregation_interval == 0:
            device = agents[0].device

            # Convert distillation states to a batch tensor (shared across all clients)
            S_d = torch.tensor(
                np.array(distill_states, dtype=np.float32), dtype=torch.float32
            ).to(device)  # shape: (|S_d|, state_dim)

            # Step 2a: Collect each client's Q-values on S_d (no-grad, inference only)
            with torch.no_grad():
                Q_clients = []  # list of (|S_d|, action_dim) tensors
                for agent in agents:
                    Q_i = agent.q_network(S_d)          # (|S_d|, action_dim)
                    Q_clients.append(Q_i)

            # Step 2b: Server computes teacher Q as the arithmetic mean (Distral)
            #   Q_teacher = (1/N) Σ_i Q_i(s)
            Q_teacher = torch.mean(torch.stack(Q_clients, dim=0), dim=0)  # (|S_d|, action_dim)
            # Soft teacher policy: π_teacher = softmax(Q_teacher / τ)
            pi_teacher = F.softmax(Q_teacher / tau, dim=-1).detach()  # (|S_d|, action_dim)

            # Step 2c: Each client distills the teacher via KL divergence
            #   L_KL = KL( π_teacher ‖ softmax(Q_i/τ) )
            #        = Σ_a π_teacher(a) * [log π_teacher(a) - log π_student(a)]
            # PyTorch F.kl_div expects log-probs as input and probs as target.
            for agent in agents:
                distill_optimizer = torch.optim.Adam(
                    agent.q_network.parameters(), lr=args.learning_rate
                )
                for _ in range(distill_steps):
                    Q_i = agent.q_network(S_d)                           # (|S_d|, action_dim)
                    log_pi_student = F.log_softmax(Q_i / tau, dim=-1)    # (|S_d|, action_dim)
                    # reduction='batchmean': sum over actions, mean over states
                    loss = F.kl_div(log_pi_student, pi_teacher, reduction='batchmean')
                    distill_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
                    distill_optimizer.step()

        # Decay exploration for all agents
        for agent in agents:
            agent.decay_exploration()

        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"{label} completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results



def run_full_comparison(episodes: int, args) -> Dict[str, ExperimentResults]:
    """
    Run all baseline comparisons for a given configuration.
    QHD and DQN methods are dispatched with their own hyperparameter sets
    (qhd_lr / dqn_lr, qhd_agg_interval / dqn_agg_interval, etc.).
    """
    all_results = {}

    print("\n" + "="*60)
    print("Q1: Performance Comparison - Homogeneous Encoders")
    print(f"  QHD: lr={args.qhd_lr}, agg={args.qhd_agg_interval}, "
          f"discount={args.qhd_discount}, eps_decay={args.qhd_exploration_decay}")
    print(f"  DQN: lr={args.dqn_lr}, agg={args.dqn_agg_interval}")
    print("="*60)

    all_results['Independent QHD']     = train_independent_qhd(episodes, args)
    all_results['FedQHD (Homogeneous)'] = train_fedqhd_homogeneous(episodes, args)
    all_results['Oracle QHD']           = train_oracle_qhd(episodes, args, use_heterogeneous=False)
    all_results['Oracle DQN']           = train_oracle_dqn(episodes, args)
    all_results['FedAvg-DQN']           = train_fedavg_dqn(episodes, args)
    # Distillation baseline with homogeneous DQN architectures (same hidden=128).
    # Federation still occurs via KL distillation in Q-space, not parameter averaging.
    all_results['Distillation FedDQN']  = train_distillation_dqn(episodes, args,
                                                                   heterogeneous=False)

    # Q2: Heterogeneous Encoders and Anchor-Based Aggregation
    if hasattr(args, 'test_heterogeneous') and args.test_heterogeneous:
        print("\n" + "="*60)
        print("Q2: Heterogeneous Encoders and Anchor-Based Aggregation")
        print("="*60)

        all_results['FedQHD (Heterogeneous)']         = train_fedqhd_heterogeneous(episodes, args)
        all_results['Oracle QHD (Heterogeneous)']     = train_oracle_qhd(episodes,  args, use_heterogeneous=True)
        all_results['Truncate FedAvg-QHD']            = train_truncate_fedavg_qhd(episodes, args)
        # Distillation baseline with heterogeneous DQN architectures [64,128,256,512].
        # Parameter averaging is undefined; only function-space distillation is applicable.
        all_results['Distillation FedDQN (Hetero)']   = train_distillation_dqn( episodes, args,
                                                                                 heterogeneous=True)

    return all_results


def save_results(results_dict: Dict[str, ExperimentResults], output_dir: str, args):
    """Save all experiment results to disk"""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual method results as JSON
    for method_name, results in results_dict.items():
        method_file = os.path.join(output_dir, f"{method_name.replace(' ', '_')}.json")
        with open(method_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

    # Save summary comparison
    summary = {
        'configuration': {
            'episodes': args.episodes,
            'num_agents': args.agent_num,
            'aggregation_interval': args.aggregation_interval,
            'learning_rate': args.learning_rate,
            'hyperdimension': args.hyperdimension,
            'environment': args.env
        },
        'methods': {}
    }

    for method_name, results in results_dict.items():
        summary['methods'][method_name] = {
            'final_avg_reward': results.final_avg_reward,
            'final_avg_success': results.final_avg_success,
            'training_time': results.training_time,
            'convergence_episode': results.convergence_episode
        }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Summary: {summary_file}")
