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


def train_independent_qhd(episodes: int, args, heterogeneous=False) -> ExperimentResults:
    """
    Baseline 1: Independent QHD (lower bound — no federation).

    Homogeneous: all clients share the same RFF encoder (args.hyperdimension).
    Heterogeneous: each client uses a different encoder dimension and bandwidth,
      mirroring the FedQHD heterogeneous setting (dims from args.hetero_dims or
      [500, 1000, 2000, 5000], bandwidth σ_i ~ Uniform[0.5σ_0, 1.5σ_0]).

    In both modes, agents are reset every qhd_agg_interval episodes — no global
    model is ever distributed, so no knowledge accumulates across rounds.
    """
    results = ExperimentResults()
    results.method_name = "Independent QHD (Heterogeneous)" if heterogeneous else "Independent QHD"

    print(f"\n{'='*60}")
    mode = "heterogeneous" if heterogeneous else "homogeneous"
    print(f"Training Independent QHD ({mode}, no federation)")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    reward_history = []
    success_history = []

    # ── Heterogeneous branch ───────────────────────────────────────────────
    if heterogeneous:
        if getattr(args, 'hetero_dims', None):
            hd_dims = [int(x) for x in args.hetero_dims.split(',')]
        else:
            hd_dims = [500, 1000, 2000, 5000]
        agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]

        def _make_hetero_agents():
            """Create fresh heterogeneous agents (resets weights and exploration)."""
            agents = []
            for i in range(num_agents):
                sigma_i = args.rff_gamma * np.random.uniform(0.5, 1.5)
                agents.append(QHDAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hd_dim=agent_dims[i],
                    rff_gamma=sigma_i,
                    learning_rate=args.qhd_lr,
                    discount_factor=args.qhd_discount,
                    exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
                    exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
                    exploration_min=getattr(args, 'qhd_exploration_min', 0.001),
                    state_bounds=state_bounds.get(args.env),
                ))
            return agents

        agents = _make_hetero_agents()

        for episode in tqdm.tqdm(range(episodes), desc="Independent QHD (Hetero)"):
            episode_rewards = []
            episode_successes = []

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

            for agent in agents:
                agent.decay_exploration()

            reward_history.append(np.mean(episode_rewards))
            success_history.append(np.mean(episode_successes))

            # Reset every K episodes — no knowledge sharing across rounds
            if (episode + 1) % args.qhd_agg_interval == 0:
                agents = _make_hetero_agents()

    # ── Homogeneous branch (original behaviour) ────────────────────────────
    else:
        fed_agent = FedAvgAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_type='qhd',
            num_agents=num_agents,
            learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
            exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
            exploration_min=getattr(args, 'qhd_exploration_min', 0.001),
            hd_dim=args.hyperdimension,
            state_bounds=state_bounds.get(args.env)
        )

        for episode in tqdm.tqdm(range(episodes), desc="Independent QHD"):
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

            fed_agent.decay_exploration()

            reward_history.append(np.mean(episode_rewards))
            success_history.append(np.mean(episode_successes))

            # Reset — discard any accumulated knowledge (no distribution)
            if (episode + 1) % args.qhd_agg_interval == 0:
                fed_agent = FedAvgAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    agent_type='qhd',
                    num_agents=num_agents,
                    learning_rate=args.qhd_lr,
                    discount_factor=args.qhd_discount,
                    exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
                    exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
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

    print(f"Independent QHD ({mode}) completed in {results.training_time:.2f}s")
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

    # Homogeneous oracle: one single centralized agent trained on all N environments.
    # Using N identical agents would be equivalent but wastes memory; one agent is cleaner
    # and avoids any LR-scaling ambiguity.
    #
    # Heterogeneous oracle: N agents are required because each client has a different
    # encoder (Φ_i); each Q_i must be compatible with its own Φ_i.  Here we do scale
    # the LR by 1/N because each agent receives N×(steps/episode) gradient updates.
    oracle_lr = args.qhd_lr / num_agents  # used only for heterogeneous branch

    # Create N agents with heterogeneous encoders (if requested)
    agents = []
    if use_heterogeneous:
        # Heterogeneous dimensions (same as FedQHD heterogeneous)
        if getattr(args, 'hetero_dims', None):
            hd_dims = [int(x) for x in args.hetero_dims.split(',')]
        else:
            hd_dims = [500, 1000, 2000, 5000]
        agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]

        for i in range(num_agents):
            # Each client has different bandwidth and dimension
            sigma_i = args.rff_gamma * np.random.uniform(0.5, 1.5)
            agent = QHDAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hd_dim=agent_dims[i],
                rff_gamma=sigma_i,
                learning_rate=oracle_lr,
                discount_factor=args.qhd_discount,
                exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
                exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
                exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
                state_bounds=state_bounds.get(args.env),
                random_seed=42 + i
            )
            agents.append(agent)
        print(f"Using heterogeneous encoders: {agent_dims}")

        # Pre-compute anchor features and Woodbury factors for hetero aggregation
        # (same anchor infrastructure as train_fedqhd_heterogeneous)
        anchor_set_size = getattr(args, 'anchor_set_size', 200)
        anchor_states_list = []
        temp_env = create_env(args)
        while len(anchor_states_list) < anchor_set_size:
            s, _ = temp_env.reset()
            anchor_states_list.append(s)
            for _ in range(50):
                a = np.random.randint(action_dim)
                ns, _, d, _, _ = temp_env.step(a)
                anchor_states_list.append(ns)
                if d or len(anchor_states_list) >= anchor_set_size:
                    break
        anchor_states_arr = np.array(anchor_states_list[:anchor_set_size])
        m_anchor = len(anchor_states_arr)
        oracle_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        oracle_anchor_features = []
        for agent in agents:
            normalized = np.array([agent.normalize_state(s) for s in anchor_states_arr])
            X_i_np = np.sqrt(2.0 / agent.hd_dim) * np.cos(normalized @ agent.omega.T + agent.b)
            oracle_anchor_features.append(torch.tensor(X_i_np, dtype=torch.float64, device=oracle_device))

        oracle_lambda_reg = 1e-4
        oracle_woodbury_factors = []
        for X_i in oracle_anchor_features:
            A_i = X_i @ X_i.T + oracle_lambda_reg * torch.eye(m_anchor, dtype=torch.float64, device=oracle_device)
            oracle_woodbury_factors.append(torch.linalg.cholesky(A_i))

        print(f"Oracle hetero anchor set: {m_anchor} states, device={oracle_device}")
        homo_n_agent_oracle = False  # hetero branch uses data-pooling (all agents on all envs)
    else:
        # Homogeneous oracle: N agents with the SAME shared encoder (identical omega/b,
        # all initialised with random_seed=42 via FedAvgAgent), each training on its OWN
        # environment (one agent per env, same as FedQHD), but aggregating EVERY episode.
        # This is the strict upper bound for FedQHD:
        #   same exploration diversity + perfect synchronisation (no K-episode lag)
        # LR is identical to FedQHD (not divided by N) because each agent still does
        # exactly one environment's worth of updates per episode.
        fed_agent_oracle = FedAvgAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_type='qhd',
            num_agents=num_agents,
            learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
            exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
            exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
            hd_dim=args.hyperdimension,
            state_bounds=state_bounds.get(args.env)
        )
        agents = fed_agent_oracle.agents  # N agents, all with identical encoder (seed=42)
        homo_n_agent_oracle = True
        print(f"Using {num_agents}-agent oracle with shared encoder + per-episode aggregation, lr={args.qhd_lr:.4f}")

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
                # Homo: agent i acts in env i (each oracle agent owns one env for acting).
                # Hetero: each env uses its own agent (different encoder).
                acting_agent = agents[env_id % len(agents)]
                action = acting_agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Update agents on this transition.
                # Hetero oracle: all agents see all envs (data pooling).
                # Homo oracle (N-agent): each agent trains only on its own env,
                # matching FedQHD structure; per-episode aggregation handles sharing.
                if homo_n_agent_oracle:
                    agents[env_id].update_model(state, action, reward, next_state, done)
                else:
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

        # Aggregation for homo oracle (per-episode = perfect synchronisation)
        if not use_heterogeneous and len(agents) > 1:
            if homo_n_agent_oracle:
                # N-agent oracle: aggregate every episode (oracle's perfect communication)
                avg_mv = np.mean([a.model_vectors for a in agents], axis=0)
                for a in agents:
                    a.model_vectors = avg_mv.copy()
            else:
                agg_interval = getattr(args, 'qhd_agg_interval', 10)
                if (episode + 1) % agg_interval == 0:
                    avg_mv = np.mean([a.model_vectors for a in agents], axis=0)
                    for a in agents:
                        a.model_vectors = avg_mv.copy()

        # Periodic anchor-based aggregation for hetero oracle (mirrors FedQHD hetero)
        if use_heterogeneous:
            agg_interval = getattr(args, 'qhd_agg_interval', 25)
            if (episode + 1) % agg_interval == 0:
                Q_list = []
                for i in range(num_agents):
                    W_i = torch.tensor(agents[i].model_vectors, dtype=torch.float64, device=oracle_device)
                    Q_list.append(oracle_anchor_features[i] @ W_i.T)  # (m, |A|)
                Q_glob_ref = torch.mean(torch.stack(Q_list, dim=0), dim=0)
                for agent_id in range(num_agents):
                    X_i = oracle_anchor_features[agent_id]
                    L_i = oracle_woodbury_factors[agent_id]
                    v = torch.linalg.solve_triangular(
                        L_i.T,
                        torch.linalg.solve_triangular(L_i, Q_glob_ref, upper=False),
                        upper=True
                    )
                    W_glob_i = X_i.T @ v  # (D_i, |A|)
                    agents[agent_id].model_vectors = W_glob_i.T.cpu().numpy()  # (|A|, D_i)

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
        exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
        exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
        exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
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
        if (episode + 1) % args.qhd_agg_interval == 0:
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
    if getattr(args, 'hetero_dims', None):
        hd_dims = [int(x) for x in args.hetero_dims.split(',')]
    else:
        hd_dims = [500, 1000, 2000, 5000]
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
            exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
            exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
            exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
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
    m = len(anchor_states)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-compute anchor feature matrices X_i for each agent.
    # The RFF encoder (omega, b) is fixed throughout training, so X_i never changes.
    # Store as torch tensors on device so per-round aggregation matmuls run on GPU.
    anchor_features = []  # list of (m, D_i) torch tensors
    for agent in agents:
        normalized = np.array([agent.normalize_state(s) for s in anchor_states])  # (m, state_dim)
        X_i_np = np.sqrt(2.0 / agent.hd_dim) * np.cos(normalized @ agent.omega.T + agent.b)
        anchor_features.append(torch.tensor(X_i_np, dtype=torch.float64, device=device))

    # Pre-compute Cholesky factors of (X_i X_i^T + λI) for the Woodbury solve.
    # These (m, m) matrices are constant — factorize once, reuse every aggregation round.
    lambda_reg = 1e-4
    woodbury_factors = []  # list of (m, m) lower-triangular torch tensors
    for X_i in anchor_features:
        A_i = X_i @ X_i.T + lambda_reg * torch.eye(m, dtype=torch.float64, device=device)
        woodbury_factors.append(torch.linalg.cholesky(A_i))

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
        if (episode + 1) % (args.qhd_agg_interval if args.qhd_agg_interval is not None else args.aggregation_interval) == 0:
            # Step 1: Each client computes Q^ref_i = X_i @ W_i^T  (Eq. 188)
            # W_i lives in numpy (local training); convert once per round.
            Q_list = []
            for i in range(num_agents):
                W_i = torch.tensor(agents[i].model_vectors, dtype=torch.float64, device=device)
                Q_list.append(anchor_features[i] @ W_i.T)          # (m, |A|)

            # Step 2: Server averages Q-values (function-space consensus)
            Q_glob_ref = torch.mean(torch.stack(Q_list, dim=0), dim=0)  # (m, |A|)

            # Step 3: Compile back to each client's parameter space (Eq. 212)
            # Woodbury:  (X^T X + λI)^{-1} X^T Q  =  X^T (X X^T + λI)^{-1} Q
            # Solved via pre-factored Cholesky L_i (m×m) — GPU-friendly triangular solves.
            for agent_id in range(num_agents):
                agent = agents[agent_id]
                X_i = anchor_features[agent_id]   # (m, D_i) on device
                L_i = woodbury_factors[agent_id]  # (m, m) lower-triangular on device

                # (X X^T + λI)^{-1} Q_glob_ref  via two triangular solves
                v = torch.linalg.solve_triangular(
                    L_i.T,
                    torch.linalg.solve_triangular(L_i, Q_glob_ref, upper=False),
                    upper=True
                )  # (m, |A|)
                W_glob_i = X_i.T @ v  # (D_i, |A|)

                # Copy back to numpy for local training
                agent.model_vectors = W_glob_i.T.cpu().numpy()  # (|A|, D_i)

                # Compute projection residual (Proposition 1, Eq. 274)
                residual = torch.linalg.norm(X_i @ W_glob_i - Q_glob_ref, ord='fro').item()
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
        learning_rate=args.dqn_lr  ,
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

        if (episode + 1) % (args.dqn_agg_interval if args.dqn_agg_interval is not None else args.aggregation_interval) == 0:
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


def train_oracle_dqn(episodes: int, args, use_heterogeneous: bool = False) -> ExperimentResults:
    """
    Baseline: Oracle DQN (Centralized DQN with Pooled Data)
    Upper bound baseline using centralized DQN training with data from all clients.
    Reference: experiments.tex line 48

    Homogeneous oracle: single centralized DQN agent trained on all N environments.
    One agent avoids LR-scaling ambiguity — full dqn_lr, no N-scaling needed.

    Heterogeneous oracle: N agents are required because each client has a different
    network architecture; each Q_i must be compatible with its own hidden size.
    LR is scaled by 1/N since each agent receives N×(steps/episode) gradient updates.
    Hidden sizes mirror train_distillation_dqn: [64, 128, 256, 512].

    Args:
        episodes: Number of training episodes
        args: Configuration arguments
        use_heterogeneous: If True, use heterogeneous DQN architectures; else homogeneous
    """
    results = ExperimentResults()
    results.method_name = "Oracle DQN (Heterogeneous)" if use_heterogeneous else "Oracle DQN"

    print(f"\n{'='*60}")
    print(f"Training Oracle DQN {'(heterogeneous-aware)' if use_heterogeneous else '(homogeneous)'}")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    oracle_lr = args.dqn_lr / num_agents  # used only for heterogeneous branch

    common_kwargs = dict(
        discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
        exploration_rate=getattr(args, 'dqn_exploration_rate', 1.0),
        exploration_decay=getattr(args, 'dqn_exploration_decay', 0.995),
        exploration_min=args.dqn_exploration_min if args.dqn_exploration_min is not None else args.exploration_min,
    )

    if use_heterogeneous:
        hidden_sizes = [64, 128, 256, 512]
        agent_hidden = [hidden_sizes[i % len(hidden_sizes)] for i in range(num_agents)]
        agents = [
            DQNAgent(state_dim=state_dim, action_dim=action_dim,
                     hidden_size=agent_hidden[i], learning_rate=oracle_lr, **common_kwargs)
            for i in range(num_agents)
        ]
        print(f"Using heterogeneous hidden sizes: {agent_hidden}")
    else:
        # Homogeneous oracle: single centralized agent trained on all N environments.
        agents = [DQNAgent(state_dim=state_dim, action_dim=action_dim,
                           learning_rate=args.dqn_lr, **common_kwargs)]

    reward_history = []
    success_history = []

    for episode in tqdm.tqdm(range(episodes), desc=f"Oracle DQN {'(Hetero)' if use_heterogeneous else '(Homo)'}"):
        episode_rewards = []
        episode_successes = []

        for env_id, env in enumerate(envs):
            state, _ = env.reset()
            done = False
            client_rewards = []

            # Homo: single centralized agent acts for all envs.
            # Hetero: each env uses its own agent (different architecture).
            acting_agent = agents[env_id] if use_heterogeneous else agents[0]

            while not done:
                action = acting_agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Train ALL agents on this transition (data pooling)
                for agent in agents:
                    agent.update_model(state, action, reward, next_state, done)

                state = next_state
                client_rewards.append(reward)

            episode_rewards.append(sum(client_rewards))
            episode_successes.append(1 if env.is_success(state) else 0)

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

    print(f"Oracle DQN completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")
    if use_heterogeneous:
        print(f"Note: Each client has architecture-compatible parameters trained on global data")

    return results


def train_truncate_fedavg_qhd(episodes: int, args,
                              heterogeneous: bool = True) -> ExperimentResults:
    """
    Baseline: Truncate/Pad FedAvg-QHD (Naive Aggregation Baseline)
    Naive approach: pad shorter vectors or truncate longer vectors to a fixed dimension.
    Reference: experiments.tex line 50

    Args:
        heterogeneous: If True, each client uses a different encoder dimension
                       [500, 1000, 2000, 5000] with randomised bandwidth (Q2 setting).
                       If False, all clients share args.hyperdimension with the same
                       bandwidth (Q1 homogeneous setting).  In the homo case
                       pad/truncate is a no-op, but the same aggregation path runs.
    """
    results = ExperimentResults()
    results.method_name = "Truncate FedAvg-QHD"

    print(f"\n{'='*60}")
    mode = "heterogeneous" if heterogeneous else "homogeneous"
    print(f"Training Truncate/Pad FedAvg-QHD ({mode})")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args) for _ in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Agent encoder dimensions
    if heterogeneous:
        if getattr(args, 'hetero_dims', None):
            hd_dims = [int(x) for x in args.hetero_dims.split(',')]
        else:
            hd_dims = [500, 1000, 2000, 5000]
        agent_dims = [hd_dims[i % len(hd_dims)] for i in range(num_agents)]
    else:
        agent_dims = [args.hyperdimension] * num_agents  # All same dim (homo)
    target_dim = args.hyperdimension  # Common dimension for averaging

    # Create agents
    agents = []
    for i in range(num_agents):
        sigma_i = args.rff_gamma * (np.random.uniform(0.5, 1.5) if heterogeneous else 1.0)
        agent = QHDAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hd_dim=agent_dims[i],
            learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
            exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
            exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
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
        if (episode + 1) % (args.qhd_agg_interval if args.qhd_agg_interval is not None else args.aggregation_interval) == 0:
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
            learning_rate=args.dqn_lr  ,
            discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
            exploration_rate=args.dqn_exploration_rate if getattr(args, 'dqn_exploration_rate', None) is not None else 1.0,
            exploration_decay=args.dqn_exploration_decay if getattr(args, 'dqn_exploration_decay', None) is not None else 0.995,
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
        if (episode + 1) % (args.dqn_agg_interval if args.dqn_agg_interval is not None else args.aggregation_interval) == 0:
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
                    agent.q_network.parameters(), lr=args.dqn_lr
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
        all_results['Oracle DQN (Heterogeneous)']     = train_oracle_dqn(episodes,  args, use_heterogeneous=True)
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
            'qhd_agg_interval': args.qhd_agg_interval if args.qhd_agg_interval is not None else args.aggregation_interval,
            'dqn_agg_interval': args.dqn_agg_interval if args.dqn_agg_interval is not None else args.aggregation_interval,
            'qhd_lr': args.qhd_lr,
            'dqn_lr': args.dqn_lr  ,
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
