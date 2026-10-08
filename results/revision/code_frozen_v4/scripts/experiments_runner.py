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


def _seed_base(args) -> int:
    """Base seed for client encoders. Encoders differ across run seeds (seed 42 reproduces the
    original encoders); in the homogeneous setting all clients share the base seed."""
    return int(getattr(args, 'random_seed', 42))


def _hetero_gamma(args, i: int) -> float:
    """Kernel parameter of heterogeneous client i: rff_gamma * U(0.5, 1.5), drawn from a
    dedicated per-seed RNG so that all heterogeneous methods share identical encoders."""
    rng = np.random.RandomState(int(getattr(args, 'random_seed', 42)) + 777)
    return float(args.rff_gamma * rng.uniform(0.5, 1.5, size=i + 1)[i])


# rl-baselines3-zoo DQN hyperparameters (hyperparams/dqn.yml, retrieved 2026-10-04).
# SB3 defaults for keys not listed there: Huber loss, max_grad_norm=10, hard target updates,
# exploration_initial_eps=1.0.
ZOO_DQN = {
    'CartPole':    dict(n_timesteps=5e4, learning_rate=2.3e-3, batch_size=64, buffer_size=100000,
                        learning_starts=1000, gamma=0.99, target_update_interval=10,
                        train_freq=256, gradient_steps=128, exploration_fraction=0.16,
                        exploration_final_eps=0.04, net_arch=[256, 256]),
    'MountainCar': dict(n_timesteps=1.2e5, learning_rate=4e-3, batch_size=128, buffer_size=10000,
                        learning_starts=1000, gamma=0.98, target_update_interval=600,
                        train_freq=16, gradient_steps=8, exploration_fraction=0.2,
                        exploration_final_eps=0.07, net_arch=[256, 256]),
    'LunarLander': dict(n_timesteps=1e5, learning_rate=6.3e-4, batch_size=128, buffer_size=50000,
                        learning_starts=0, gamma=0.99, target_update_interval=250,
                        train_freq=4, gradient_steps=-1, exploration_fraction=0.12,
                        exploration_final_eps=0.1, net_arch=[256, 256]),
    'Acrobot':     dict(n_timesteps=1e5, learning_rate=6.3e-4, batch_size=128, buffer_size=50000,
                        learning_starts=0, gamma=0.99, target_update_interval=250,
                        train_freq=4, gradient_steps=-1, exploration_fraction=0.12,
                        exploration_final_eps=0.1, net_arch=[256, 256]),
}


def dqn_config(args, hidden_size=None, lr_scale: float = 1.0) -> dict:
    """Constructor kwargs for every DQN client/baseline.

    ``--dqn_preset zoo`` uses the rl-baselines3-zoo hyperparameters for the environment;
    heterogeneous clients keep two hidden layers but change their width (``hidden_size``).
    ``--dqn_preset legacy`` (default) reproduces the submitted configuration.
    """
    if getattr(args, 'dqn_preset', 'legacy') == 'zoo':
        z = ZOO_DQN[args.env]
        width = hidden_size or z['net_arch'][0]
        # pooled-data references see N× the transitions: --dqn_lr_scale and
        # --dqn_eps_steps_scale (ε-schedule length) are tuned for them (default 1 = zoo)
        lr_scale = lr_scale * float(getattr(args, 'dqn_lr_scale', None) or 1.0)
        eps_steps = z['n_timesteps'] * float(getattr(args, 'dqn_eps_steps_scale', None) or 1.0)
        dev = getattr(args, 'dqn_device', None)
        extra = {'device': torch.device(dev)} if dev and dev != 'auto' else {}
        return dict(
            **extra,
            learning_rate=z['learning_rate'] * lr_scale, discount_factor=z['gamma'],
            batch_size=z['batch_size'], buffer_size=z['buffer_size'],
            learning_starts=z['learning_starts'], train_freq=z['train_freq'],
            gradient_steps=z['gradient_steps'], target_update_interval=z['target_update_interval'],
            loss='huber', max_grad_norm=10.0, net_arch=[width] * len(z['net_arch']),
            eps_schedule=dict(initial=1.0, final=z['exploration_final_eps'],
                              fraction=z['exploration_fraction'], total_steps=eps_steps),
        )
    kw = dict(
        learning_rate=args.dqn_lr * lr_scale,
        discount_factor=args.dqn_discount if args.dqn_discount is not None else args.discount_factor,
        exploration_rate=getattr(args, 'dqn_exploration_rate', None) or 1.0,
        exploration_decay=getattr(args, 'dqn_exploration_decay', None) or 0.995,
        exploration_min=(args.dqn_exploration_min if getattr(args, 'dqn_exploration_min', None)
                         is not None else 0.01),
    )
    if hidden_size:
        kw['hidden_size'] = hidden_size
    return kw


def _td_done(env, done: bool, args) -> bool:
    """Episode-end flag used in the TD target.

    By default (legacy behaviour, used for the submitted results) time-limit truncation is
    treated as terminal. With ``--bootstrap_on_truncation`` only true terminations stop
    bootstrapping, which is the correct target for time-limited tasks.
    """
    if getattr(args, 'bootstrap_on_truncation', False) and hasattr(env, 'last_terminated'):
        return env.last_terminated
    return done



def collect_anchor_states(args, action_dim: int, m: int) -> np.ndarray:
    """Server anchor set S_ref: m states from uniformly random-action rollouts.

    ``--anchor_source nominal`` (default) rolls out a server-side copy of the nominal
    environment. ``--anchor_source clients`` pools equal shares of random rollouts from
    every client's environment (relevant under dynamics heterogeneity, where client state
    distributions differ).
    """
    source = getattr(args, 'anchor_source', 'nominal')
    if source in ('uniform', 'mix'):
        # Uniform anchors over the normalized state box (no simulator or rollouts needed);
        # 'mix' = half uniform, half nominal random-rollout anchors.
        n_uni = m if source == 'uniform' else m // 2
        uni = sample_uniform_states(args.env, n_uni,
                                    int(getattr(args, 'random_seed', 42)) + 54321)
        if source == 'uniform':
            return uni
        roll = collect_anchor_states(_with(args, anchor_source='nominal'), action_dim, m - n_uni)
        return np.concatenate([roll, uni])
    if source == 'clients':
        n = args.agent_num
        sources = [create_env(args, client_id=i) for i in range(n)]
        quotas = [m // n + (1 if i < m % n else 0) for i in range(n)]
    else:
        sources, quotas = [create_env(args)], [m]
    # Dedicated RNG: every method with the same seed gets the same anchor/query set.
    rng = np.random.RandomState(int(getattr(args, 'random_seed', 42)) + 12345)
    anchor_states = []
    for temp_env, quota in zip(sources, quotas):
        got = []
        while len(got) < quota:
            state, _ = temp_env.reset()
            got.append(state)
            for _ in range(50):  # Random rollout steps
                action = rng.randint(action_dim)
                next_state, _, done, _, _ = temp_env.step(action)
                got.append(next_state)
                if done or len(got) >= quota:
                    break
        anchor_states.extend(got[:quota])
    return np.array(anchor_states[:m])


def _with(args, **kw):
    a = copy.copy(args)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def sample_uniform_states(env_name: str, n: int, seed: int) -> np.ndarray:
    """n states uniform over the normalized state box [-1, 1]^d (mapped back to raw states).

    Structured coordinates stay valid: Acrobot (cos, sin) pairs come from uniform angles, and
    LunarLander's leg-contact flags are sampled from {0, 1}.
    """
    rng = np.random.RandomState(seed)
    b = np.asarray(state_bounds[env_name], dtype=float)
    S = b[:, 0] + rng.random_sample((n, len(b))) * (b[:, 1] - b[:, 0])
    if env_name == 'Acrobot':
        t1, t2 = rng.uniform(-np.pi, np.pi, (2, n))
        S[:, 0], S[:, 1], S[:, 2], S[:, 3] = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
    if env_name == 'LunarLander':
        S[:, 6:8] = rng.randint(0, 2, (n, 2))
    return S.astype(np.float32)


def ridge_compile(factor, X_i: torch.Tensor, Q_glob_ref: torch.Tensor) -> torch.Tensor:
    """W = (X^T X + λI)^{-1} X^T Q, using the pre-factored dual (m x m) or primal (D x D) form."""
    form, L = factor
    if form == 'dual':
        v = torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, Q_glob_ref, upper=False), upper=True)
        return X_i.T @ v
    rhs = X_i.T @ Q_glob_ref
    return torch.linalg.solve_triangular(
        L.T, torch.linalg.solve_triangular(L, rhs, upper=False), upper=True)


def compile_teacher(factor, X_i: torch.Tensor, Q_glob_ref: torch.Tensor, agent, args) -> torch.Tensor:
    """Compile the anchor teacher into client i's readout.

    ``--compile_mode overwrite`` (Algorithm 2 as submitted):
        W = argmin ||X W − Q||² + λ||W||²            = (XᵀX + λI)⁻¹ Xᵀ Q
    ``--compile_mode warmstart``:
        W = argmin ||X W − Q||² + λ||W − W_local||²  = W_local + (XᵀX + λI)⁻¹ Xᵀ (Q − X W_local)
    The warm start keeps the component of the local readout outside the anchor span (it is
    only shrunk inside it), so local knowledge on states not covered by the anchors survives.
    """
    if getattr(args, 'compile_mode', 'overwrite') == 'warmstart':
        W_loc = torch.tensor(agent.model_vectors.T, dtype=X_i.dtype, device=X_i.device)
        return W_loc + ridge_compile(factor, X_i, Q_glob_ref - X_i @ W_loc)
    return ridge_compile(factor, X_i, Q_glob_ref)


def compile_diagnostics_init(anchor_features, lambda_reg: float) -> dict:
    """Static anchor-conditioning quantities of Theorem 2 for each client.

    One SVD per client; the row-space basis is kept in host memory (numpy) and the GPU
    workspace is released immediately, so diagnostics do not limit how many runs share a GPU.
    """
    diag = {'lambda': lambda_reg, 'm': int(anchor_features[0].shape[0]), 'clients': [],
            'rounds': [], '_row_bases': []}
    for X_i in anchor_features:
        _, sv, Vh = torch.linalg.svd(X_i, full_matrices=False)
        tol = sv.max() * max(X_i.shape) * torch.finfo(sv.dtype).eps
        keep = sv > tol
        pos = sv[keep]
        gamma = float(pos.min() ** 2)
        s2 = (sv ** 2).cpu().numpy()
        diag['clients'].append({
            'D': int(X_i.shape[1]), 'rank': int(pos.numel()), 'gamma_min': gamma,
            'amplification': float(np.sqrt(X_i.shape[0] / (gamma + lambda_reg))),
            'shrinkage_factor': float(lambda_reg / (gamma + lambda_reg)),
            # directions compiled with less than 50% ridge shrinkage (σ² > λ)
            'effective_rank_lambda': int((s2 > lambda_reg).sum()),
            'sv2_quantiles': [float(q) for q in np.quantile(s2, [0.0, 0.1, 0.5, 0.9, 1.0])],
        })
        diag['_row_bases'].append(Vh[keep].cpu().numpy())  # (rank, D_i) orthonormal rows
        del sv, Vh, pos, keep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diag


def compile_diagnostics_round(diag: dict, agents, anchor_features, Q_list, Q_glob_ref,
                              visited, episode: int):
    """Per-round diagnostics: teacher disagreement and coverage ρ_i on visited states."""
    rec = {'episode': int(episode + 1), 'teacher_disagreement': [], 'rho_mean': [],
           'rho_max': []}
    for i, agent in enumerate(agents):
        rec['teacher_disagreement'].append(
            float(torch.sqrt(torch.mean((Q_list[i] - Q_glob_ref) ** 2)).item()))
        if visited[i]:
            S = np.array([agent.normalize_state(s) for s in visited[i]])
            Phi = np.sqrt(2.0 / agent.hd_dim) * np.cos(S @ agent.omega.T + agent.b)
            proj = Phi @ diag['_row_bases'][i].T
            nrm = (Phi ** 2).sum(1)
            # ρ normalised by ||Φ||²: relative out-of-span feature energy in [0, 1]
            rel = np.sqrt(np.clip(nrm - (proj ** 2).sum(1), 0.0, None) / nrm)
            rec['rho_mean'].append(float(rel.mean()))
            rec['rho_max'].append(float(rel.max()))
    diag['rounds'].append(rec)


def finalize_diagnostics(diag: dict) -> dict:
    """Drop non-serializable entries before saving."""
    return {k: v for k, v in diag.items() if not k.startswith('_')}


def greedy_action(agent, state) -> int:
    """Greedy (ε = 0) action of any client agent."""
    if hasattr(agent, 'greedy_action'):
        return agent.greedy_action(state)
    if isinstance(agent, QHDAgent):
        return int(np.argmax(agent.model_vectors @ agent.encode_state(state)))
    with torch.no_grad():
        q = agent.q_network(agent._state_to_tensor(state).unsqueeze(0))
    return int(torch.argmax(q).item())


def greedy_evaluate(agents, args, n_episodes: int = None) -> dict:
    """Return of each client's greedy policy on fresh copies of its own environment.

    Client i is evaluated with agents[i] (agents[0] for a single centralized agent), on
    ``n_episodes`` episodes whose initial states come from a dedicated evaluation seed.
    """
    n_episodes = n_episodes or int(getattr(args, 'eval_episodes', 10) or 10)
    per_client = []
    for i in range(args.agent_num):
        agent = agents[i] if len(agents) > 1 else agents[0]
        env = create_env(args, client_id=i)
        try:
            env.env.reset(seed=10 ** 6 + int(getattr(args, 'random_seed', 42)) * 1000 + i)
        except TypeError:
            pass
        rets = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            done, total = False, 0.0
            while not done:
                state, reward, done, _, _ = env.step(greedy_action(agent, state))
                total += reward
            rets.append(total)
        per_client.append(float(np.mean(rets)))
    return {'eval_return': float(np.mean(per_client)), 'eval_per_client': per_client,
            'eval_episodes': n_episodes}


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
        self.diagnostics = {}  # Optional per-method diagnostics (e.g. compile conditioning)
        self.evaluation = {}   # Greedy-policy evaluation at the end of training

    def to_dict(self):
        return {
            'method_name': self.method_name,
            'reward_history': self.reward_history,
            'success_history': self.success_history,
            'training_time': self.training_time,
            'convergence_episode': self.convergence_episode,
            'projection_residuals': self.projection_residuals,
            'final_avg_reward': self.final_avg_reward,
            'final_avg_success': self.final_avg_success,
            'diagnostics': self.diagnostics,
            'evaluation': self.evaluation
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
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
                sigma_i = _hetero_gamma(args, i)
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
                    random_seed=_seed_base(args) + i,
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
                    agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
                    state = next_state
                    total_reward += reward

                episode_rewards.append(total_reward)
                episode_successes.append(1 if env.is_success(state) else 0)

            for agent in agents:
                agent.decay_exploration()

            reward_history.append(np.mean(episode_rewards))
            success_history.append(np.mean(episode_successes))

            # Reset every K episodes — no knowledge sharing across rounds (legacy behaviour;
            # --indep_no_reset gives true independent learning without re-initialisation)
            if (episode + 1) % args.qhd_agg_interval == 0 and not getattr(args, 'indep_no_reset', False):
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
            state_bounds=state_bounds.get(args.env),
            random_seed=_seed_base(args)
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
                    agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
                    state = next_state
                    total_reward += reward

                episode_rewards.append(total_reward)
                episode_successes.append(1 if env.is_success(state) else 0)

            fed_agent.decay_exploration()

            reward_history.append(np.mean(episode_rewards))
            success_history.append(np.mean(episode_successes))

            # Reset — discard any accumulated knowledge (no distribution); legacy behaviour,
            # disabled by --indep_no_reset
            if (episode + 1) % args.qhd_agg_interval == 0 and not getattr(args, 'indep_no_reset', False):
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
                    state_bounds=state_bounds.get(args.env),
                    random_seed=_seed_base(args)
                )

    end_time = time.time()

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents if heterogeneous else fed_agent.agents, args)
    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"Independent QHD ({mode}) completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")

    return results


def train_oracle_qhd(episodes: int, args, use_heterogeneous: bool = False,
                     pooled: bool = False) -> ExperimentResults:
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
    results.method_name = ("Oracle QHD (Heterogeneous)" if use_heterogeneous else
                           "Pooled QHD" if pooled else "Oracle QHD")

    print(f"\n{'='*60}")
    print(f"Training Oracle QHD {'(heterogeneous-aware)' if use_heterogeneous else '(homogeneous)'}")
    print("Server: Trains N separate Q-functions on pooled data")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
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
            sigma_i = _hetero_gamma(args, i)
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
                random_seed=_seed_base(args) + i
            )
            agents.append(agent)
        print(f"Using heterogeneous encoders: {agent_dims}")

        # Pre-compute anchor features and Woodbury factors for hetero aggregation
        # (same anchor infrastructure as train_fedqhd_heterogeneous)
        anchor_set_size = getattr(args, 'anchor_set_size', 200)
        anchor_states_arr = collect_anchor_states(args, action_dim, anchor_set_size)
        m_anchor = len(anchor_states_arr)
        oracle_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        oracle_anchor_features = []
        for agent in agents:
            normalized = np.array([agent.normalize_state(s) for s in anchor_states_arr])
            X_i_np = np.sqrt(2.0 / agent.hd_dim) * np.cos(normalized @ agent.omega.T + agent.b)
            oracle_anchor_features.append(torch.tensor(X_i_np, dtype=torch.float64, device=oracle_device))

        oracle_lambda_reg = float(getattr(args, 'ridge_lambda', None) or 1e-4)
        oracle_ridge_factors = []  # same dual/primal choice as train_fedqhd_heterogeneous
        for X_i in oracle_anchor_features:
            D_i = X_i.shape[1]
            if m_anchor <= D_i:
                A_i = X_i @ X_i.T + oracle_lambda_reg * torch.eye(m_anchor, dtype=torch.float64, device=oracle_device)
                oracle_ridge_factors.append(('dual', torch.linalg.cholesky(A_i)))
            else:
                A_i = X_i.T @ X_i + oracle_lambda_reg * torch.eye(D_i, dtype=torch.float64, device=oracle_device)
                oracle_ridge_factors.append(('primal', torch.linalg.cholesky(A_i)))

        print(f"Oracle hetero anchor set: {m_anchor} states, device={oracle_device}")
        homo_n_agent_oracle = False  # hetero branch uses data-pooling (all agents on all envs)
    elif pooled:
        # Centralized oracle: ONE QHD agent (shared encoder) acts in every client environment
        # and is trained on all clients' transitions (true pooled-data reference).
        agents = [QHDAgent(
            state_dim=state_dim, action_dim=action_dim, hd_dim=args.hyperdimension,
            rff_gamma=args.rff_gamma, learning_rate=args.qhd_lr,
            discount_factor=args.qhd_discount,
            exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
            exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
            exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
            state_bounds=state_bounds.get(args.env), random_seed=_seed_base(args))]
        homo_n_agent_oracle = False
        print(f"Using pooled single-agent oracle, lr={args.qhd_lr:.4f}")
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
            state_bounds=state_bounds.get(args.env),
            random_seed=_seed_base(args)
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
                    agents[env_id].update_model(state, action, reward, next_state, _td_done(env, done, args))
                else:
                    for agent in agents:
                        agent.update_model(state, action, reward, next_state, _td_done(env, done, args))

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
                    W_glob_i = compile_teacher(oracle_ridge_factors[agent_id], X_i, Q_glob_ref,
                                               agents[agent_id], args)
                    agents[agent_id].model_vectors = W_glob_i.T.cpu().numpy()  # (|A|, D_i)

        # Average across all client environments
        reward_history.append(np.mean(episode_rewards))
        success_history.append(np.mean(episode_successes))


    end_time = time.time()

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]

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
        state_bounds=state_bounds.get(args.env),
        random_seed=_seed_base(args)
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
                agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(fed_agent.agents, args)
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
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
        sigma_i = _hetero_gamma(args, i)  # Heterogeneous bandwidth
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
            random_seed=_seed_base(args) + i  # Different seed per agent
        )
        agents.append(agent)

    # Server: Construct anchor set (methodology.tex line 36)
    # Collect m=args.anchor_set_size states from random rollouts
    anchor_states = collect_anchor_states(args, action_dim, args.anchor_set_size)
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

    # Pre-compute Cholesky factors for the ridge solve. These matrices are constant —
    # factorize once, reuse every aggregation round. When m <= D_i we use the Woodbury
    # (m x m) form X^T (X X^T + λI)^{-1} Q; when m > D_i the primal (D_i x D_i) form
    # (X^T X + λI)^{-1} X^T Q, which is cheaper and better conditioned.
    lambda_reg = float(getattr(args, 'ridge_lambda', None) or 1e-4)
    ridge_factors = []  # list of (form, lower-triangular factor)
    for X_i in anchor_features:
        D_i = X_i.shape[1]
        if m <= D_i:
            A_i = X_i @ X_i.T + lambda_reg * torch.eye(m, dtype=torch.float64, device=device)
            ridge_factors.append(('dual', torch.linalg.cholesky(A_i)))
        else:
            A_i = X_i.T @ X_i + lambda_reg * torch.eye(D_i, dtype=torch.float64, device=device)
            ridge_factors.append(('primal', torch.linalg.cholesky(A_i)))

    # Diagnostics for Theorem 2: anchor conditioning γ_i, rank, and the coverage term ρ_i(s)
    # evaluated on recently visited client states (right singular basis of X_i).
    no_diag = bool(getattr(args, 'no_diagnostics', False))  # e.g. tuning runs
    compile_diag = None if no_diag else compile_diagnostics_init(anchor_features, lambda_reg)
    visited = [[] for _ in range(num_agents)]  # recent visited states per client

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
                agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
                visited[agent_id].append(state)
                state = next_state
                total_reward += reward
            visited[agent_id] = visited[agent_id][-512:]

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

            if compile_diag is not None:
                compile_diagnostics_round(compile_diag, agents, anchor_features, Q_list,
                                          Q_glob_ref, visited, episode)

            # Step 3: Compile back to each client's parameter space (Eq. 212)
            # Woodbury:  (X^T X + λI)^{-1} X^T Q  =  X^T (X X^T + λI)^{-1} Q
            # Solved via pre-factored Cholesky factors — GPU-friendly triangular solves.
            for agent_id in range(num_agents):
                agent = agents[agent_id]
                X_i = anchor_features[agent_id]   # (m, D_i) on device
                W_glob_i = compile_teacher(ridge_factors[agent_id], X_i, Q_glob_ref, agent, args)  # (D_i, |A|)

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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
    results.reward_history = reward_history
    results.success_history = success_history
    results.training_time = end_time - start_time
    results.projection_residuals = projection_residuals
    results.diagnostics = finalize_diagnostics(compile_diag) if compile_diag else {}
    results.final_avg_reward = np.mean(reward_history[-100:])
    results.final_avg_success = np.mean(success_history[-100:])

    print(f"FedQHD (Heterogeneous) completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")
    print(f"Avg projection residual: {np.mean(projection_residuals):.4f}")

    return results


def train_fedavg_dqn(episodes: int, args, prox: bool = False) -> ExperimentResults:
    """
    Baseline 3: FedAvg-DQN
    Federated deep Q-learning with parameter averaging.
    Uses a 2-layer MLP with 128 hidden units per layer.
    """
    results = ExperimentResults()
    results.method_name = "FedProx-DQN" if prox else "FedAvg-DQN"

    print(f"\n{'='*60}")
    print(f"Training {results.method_name}")
    print(f"{'='*60}")

    start_time = time.time()

    num_agents = args.agent_num
    envs = [create_env(args, client_id=i) for i in range(num_agents)]

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
        state_bounds=state_bounds.get(args.env),
        random_seed=_seed_base(args),
        dqn_kwargs=dqn_config(args) if getattr(args, 'dqn_preset', 'legacy') == 'zoo' else None,
    )
    if prox:
        mu = float(getattr(args, 'fedprox_mu', None) or 0.01)
        for agent in fed_agent.agents:
            agent.prox_mu = mu
            agent.set_prox_reference()
        results.diagnostics = {'fedprox_mu': mu}

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
                agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(fed_agent.agents, args)
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_dim

    # Legacy: heterogeneous oracle LR is scaled by 1/N (each agent sees N× the transitions).
    # Zoo preset: the zoo learning rate is used unchanged.
    legacy = getattr(args, 'dqn_preset', 'legacy') != 'zoo'
    oracle_scale = 1.0 / num_agents if legacy else 1.0

    if use_heterogeneous:
        hidden_sizes = [64, 128, 256, 512]
        agent_hidden = [hidden_sizes[i % len(hidden_sizes)] for i in range(num_agents)]
        agents = [
            DQNAgent(state_dim=state_dim, action_dim=action_dim,
                     **dqn_config(args, hidden_size=agent_hidden[i], lr_scale=oracle_scale))
            for i in range(num_agents)
        ]
        print(f"Using heterogeneous hidden sizes: {agent_hidden}")
    else:
        # Homogeneous oracle: single centralized agent trained on all N environments.
        agents = [DQNAgent(state_dim=state_dim, action_dim=action_dim, **dqn_config(args))]

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
                    agent.update_model(state, action, reward, next_state, _td_done(env, done, args))

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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
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
        sigma_i = _hetero_gamma(args, i) if heterogeneous else args.rff_gamma
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
            rff_gamma=sigma_i,
            random_seed=_seed_base(args) + (i if heterogeneous else 0)
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
                agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
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
    envs = [create_env(args, client_id=i) for i in range(num_agents)]
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
            **dqn_config(args, hidden_size=agent_hidden[i] if heterogeneous else None),
        )
        agents.append(agent)

    # Build distillation set S_d: the same random-rollout anchor set S_ref used by FedQHD
    # (same size m, same states for the same seed).
    distill_states = list(collect_anchor_states(args, action_dim, args.anchor_set_size))

    # Distillation temperature τ: higher → softer targets (less confident teacher)
    tau = float(getattr(args, 'distill_tau', None) or 1.0)
    # Number of gradient steps per distillation round
    distill_steps = int(getattr(args, 'distill_steps', None) or 10)
    # Loss: 'kl' = policy distillation on softmax(Q/τ) (default, as submitted);
    #       'mse' = Q-value regression to the teacher (the same target FedQHD compiles).
    distill_loss = getattr(args, 'distill_loss', None) or 'kl'
    # Federation interval: by default the DQN interval; --distill_interval overrides it
    # (set it to qhd_agg_interval for a communication-matched comparison).
    distill_interval = (getattr(args, 'distill_interval', None) or
                        (args.dqn_agg_interval if args.dqn_agg_interval is not None
                         else args.aggregation_interval))

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
                agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_successes.append(1 if env.is_success(state) else 0)

        # ── Phase 2: Distillation aggregation every K episodes ───────────────
        if (episode + 1) % distill_interval == 0:
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
                    agent.q_network.parameters(), lr=agent.lr
                )
                for _ in range(distill_steps):
                    Q_i = agent.q_network(S_d)                           # (|S_d|, action_dim)
                    if distill_loss == 'mse':
                        loss = F.mse_loss(Q_i, Q_teacher.detach())
                    else:
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

    if int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
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
