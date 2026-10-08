"""
Additional baselines for the post-review revision of the FedQHD paper.

All methods use the same heterogeneous client setup as ``train_fedqhd_heterogeneous``
(same encoder dimensions, kernel parameters, encoder seeds, environments and, where
applicable, the same anchor set for a given ``--random_seed``), so differences come only
from the federation step.

* ``train_fedhql``     — FedHQL (Fan et al., 2023): server-side rollouts on a separate copy of
                         the MDP, FedUCB action selection a = argmax_a Qbar + λ_ucb·Q_std,
                         FedTD update of the aggregated value, and κ regression steps per client
                         on (s, a, Qbar_new). Clients are heterogeneous QHD agents (controlled
                         comparison with FedQHD) or heterogeneous DQNs (as in the FedHQL paper).
* ``train_fedqhd_gd``  — FedQHD-GD: identical to heterogeneous FedQHD (same anchors, teacher and
                         interval) but the closed-form ridge compile is replaced by κ gradient
                         steps on the same ridge objective, warm-started from the local W_i.
                         Isolates "closed-form vs. iterative" compilation.
"""

import time
from typing import List

import numpy as np
import torch
import tqdm

from env.utils import create_env
from env.bounds import state_bounds
from agent.qhd_agent import QHDAgent
from agent.dqn_agent import DQNAgent
from experiments_runner import (  # scripts/ is on sys.path (see run_single_method.py)
    ExperimentResults, collect_anchor_states, _seed_base, _hetero_gamma, _td_done, dqn_config,
    greedy_evaluate,
)


# ── client construction ─────────────────────────────────────────────────────────

def _hetero_dims(args, num_agents: int) -> List[int]:
    if getattr(args, 'hetero_dims', None):
        hd_dims = [int(x) for x in args.hetero_dims.split(',')]
    else:
        hd_dims = [500, 1000, 2000, 5000]
    return [hd_dims[i % len(hd_dims)] for i in range(num_agents)]


def _make_qhd_clients(args, state_dim: int, action_dim: int) -> List[QHDAgent]:
    """Heterogeneous QHD clients, identical to those of train_fedqhd_heterogeneous."""
    dims = _hetero_dims(args, args.agent_num)
    return [QHDAgent(
        state_dim=state_dim, action_dim=action_dim, hd_dim=dims[i],
        rff_gamma=_hetero_gamma(args, i),
        learning_rate=args.qhd_lr, discount_factor=args.qhd_discount,
        exploration_rate=(getattr(args, 'qhd_exploration_rate', None) or 1.0),
        exploration_decay=(getattr(args, 'qhd_exploration_decay', None) or 0.995),
        exploration_min=getattr(args, 'qhd_exploration_min', 0.01),
        state_bounds=state_bounds.get(args.env),
        random_seed=_seed_base(args) + i,
    ) for i in range(args.agent_num)]


def _make_dqn_clients(args, state_dim: int, action_dim: int) -> List[DQNAgent]:
    """Heterogeneous DQN clients, identical to those of train_distillation_dqn(hetero)."""
    hidden_sizes = [64, 128, 256, 512]
    return [DQNAgent(state_dim=state_dim, action_dim=action_dim,
                     **dqn_config(args, hidden_size=hidden_sizes[i % len(hidden_sizes)]))
            for i in range(args.agent_num)]


def _features(agent: QHDAgent, states: np.ndarray) -> np.ndarray:
    S = np.array([agent.normalize_state(s) for s in states])
    return np.sqrt(2.0 / agent.hd_dim) * np.cos(S @ agent.omega.T + agent.b)


# ── uniform client interface (Q-values on a batch, regression on (s, a, y)) ──────

def _q_batch(agent, states: np.ndarray) -> np.ndarray:
    if isinstance(agent, QHDAgent):
        return _features(agent, states) @ agent.model_vectors.T
    with torch.no_grad():
        S = torch.tensor(np.asarray(states, dtype=np.float32), device=agent.device)
        return agent.q_network(S).cpu().numpy()


def _regress(agent, states: np.ndarray, actions: np.ndarray, targets: np.ndarray,
             steps: int, lr: float):
    """κ gradient steps on 1/B Σ (y_b − Q(s_b, a_b))² (FedHQL Eq. 15–16)."""
    if isinstance(agent, QHDAgent):
        Phi = _features(agent, states)                     # (B, D)
        onehot = np.eye(agent.action_dim)[actions]         # (B, |A|)
        for _ in range(steps):
            q = np.sum((Phi @ agent.model_vectors.T) * onehot, axis=1)
            err = (targets - q)[:, None] * onehot          # (B, |A|)
            agent.model_vectors += lr * (err.T @ Phi) / len(states)
        return
    S = torch.tensor(np.asarray(states, dtype=np.float32), device=agent.device)
    A = torch.tensor(actions, dtype=torch.long, device=agent.device)
    Y = torch.tensor(targets, dtype=torch.float32, device=agent.device)
    opt = torch.optim.Adam(agent.q_network.parameters(), lr=lr)
    for _ in range(steps):
        q = agent.q_network(S).gather(1, A[:, None]).squeeze(1)
        loss = torch.mean((Y - q) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
        opt.step()


def _local_episode(agent, env, args):
    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update_model(state, action, reward, next_state, _td_done(env, done, args))
        state = next_state
        total += reward
    return total, 1 if env.is_success(state) else 0


def _finish(results, rewards, successes, start, diagnostics, agents=None, args=None):
    results.training_time = time.time() - start  # excludes the evaluation below
    if agents is not None and int(getattr(args, 'eval_episodes', 10) or 0) > 0:
        results.evaluation = greedy_evaluate(agents, args)
    results.reward_history = rewards
    results.success_history = successes
    results.final_avg_reward = float(np.mean(rewards[-100:]))
    results.final_avg_success = float(np.mean(successes[-100:]))
    results.diagnostics = diagnostics
    print(f"{results.method_name} completed in {results.training_time:.2f}s")
    print(f"Final avg reward (last 100 eps): {results.final_avg_reward:.2f}")
    return results


# ── FedHQL ──────────────────────────────────────────────────────────────────────

def train_fedhql(episodes: int, args, client_type: str = 'qhd') -> ExperimentResults:
    """FedHQL (Fan et al., 2023) with heterogeneous clients.

    Every K episodes the server runs ``fedhql_batch`` parallel rollouts of horizon
    ``fedhql_horizon`` on its own copy of the (nominal) environment. At each server step:
      Qbar = mean_n Q_n(s,·),  Q_std = std_n Q_n(s,·)                    (Eq. 10–11)
      a = argmax_a Qbar(s,a) + λ_ucb · Q_std(s,a)                         (FedUCB, Eq. 12–13)
      Qbar_new(s,a) = Qbar(s,a) + α_s (r + γ max_b Qbar(s',b) − Qbar(s,a)) (FedTD, Eq. 14)
    and every client then performs κ regression steps towards Qbar_new on the collected
    (s, a) pairs (Eq. 15–16). Defaults follow the FedHQL paper (κ=64, H=16, α_s=0.05,
    batch 128, λ_ucb=1). Server environment steps are reported in the diagnostics.
    """
    results = ExperimentResults()
    results.method_name = f"FedHQL ({client_type.upper()} clients)"
    print(f"\n{'=' * 60}\nTraining {results.method_name}\n{'=' * 60}")
    start = time.time()

    envs = [create_env(args, client_id=i) for i in range(args.agent_num)]
    state_dim, action_dim = envs[0].state_dim, envs[0].action_dim
    agents = (_make_qhd_clients if client_type == 'qhd' else _make_dqn_clients)(
        args, state_dim, action_dim)

    K = args.qhd_agg_interval if client_type == 'qhd' else args.dqn_agg_interval
    K = getattr(args, 'fedhql_interval', None) or K
    kappa = int(getattr(args, 'fedhql_kappa', None) or 64)
    horizon = int(getattr(args, 'fedhql_horizon', None) or 16)
    batch = int(getattr(args, 'fedhql_batch', None) or 128)
    alpha_s = float(getattr(args, 'fedhql_alpha', None) or 0.05)
    lam_ucb = float(getattr(args, 'fedhql_ucb', None) if getattr(args, 'fedhql_ucb', None)
                    is not None else 1.0)
    gamma = args.qhd_discount if client_type == 'qhd' else agents[0].gamma
    reg_lr = getattr(args, 'fedhql_lr', None) or (0.5 if client_type == 'qhd' else agents[0].lr)

    server_envs = [create_env(args) for _ in range(batch)]
    server_steps = 0
    rewards, successes = [], []

    for episode in tqdm.tqdm(range(episodes), desc=results.method_name):
        out = [_local_episode(a, e, args) for a, e in zip(agents, envs)]
        rewards.append(float(np.mean([o[0] for o in out])))
        successes.append(float(np.mean([o[1] for o in out])))

        if (episode + 1) % K == 0:
            S_buf, A_buf, Y_buf = [], [], []
            states = np.array([e.reset()[0] for e in server_envs])
            alive = np.ones(batch, dtype=bool)
            for _ in range(horizon):
                idx = np.where(alive)[0]
                if idx.size == 0:
                    break
                Qs = np.stack([_q_batch(a, states[idx]) for a in agents])     # (N, B, |A|)
                q_bar, q_std = Qs.mean(0), Qs.std(0)
                acts = np.argmax(q_bar + lam_ucb * q_std, axis=1)
                next_states, rews, terms = [], [], []
                for j, b in enumerate(idx):
                    ns, r, d, tr, _ = server_envs[b].step(int(acts[j]))
                    next_states.append(ns); rews.append(r)
                    terms.append(server_envs[b].last_terminated)
                    if d:
                        alive[b] = False
                server_steps += len(idx)
                next_states = np.array(next_states)
                q_next = np.stack([_q_batch(a, next_states) for a in agents]).mean(0)
                q_sa = q_bar[np.arange(len(idx)), acts]
                td_target = np.array(rews) + gamma * (1 - np.array(terms)) * q_next.max(1)
                S_buf.append(states[idx]); A_buf.append(acts)
                Y_buf.append(q_sa + alpha_s * (td_target - q_sa))
                states = states.copy()
                states[idx] = next_states
            S_all, A_all, Y_all = np.concatenate(S_buf), np.concatenate(A_buf), np.concatenate(Y_buf)
            for a in agents:
                _regress(a, S_all, A_all, Y_all, kappa, reg_lr)

        for a in agents:
            a.decay_exploration()

    diag = {'server_env_steps': int(server_steps), 'kappa': kappa, 'horizon': horizon,
            'batch': batch, 'alpha_s': alpha_s, 'lambda_ucb': lam_ucb, 'interval': K,
            'regression_lr': reg_lr}
    return _finish(results, rewards, successes, start, diag, agents, args)


# ── FedQHD-GD: iterative compilation of the same anchor teacher ─────────────────

def train_fedqhd_gd(episodes: int, args) -> ExperimentResults:
    """Heterogeneous FedQHD with the closed-form compile replaced by gradient descent.

    Each round, client i takes ``gd_steps`` full-batch gradient steps on
    (1/m) ||X_i W − Q_glob_ref||_F² + (λ/m) ||W||_F², warm-started from its local W_i, with
    step size ``gd_lr``. Anchors, teacher, interval and clients are identical to FedQHD.
    """
    results = ExperimentResults()
    steps = int(getattr(args, 'gd_steps', None) or 10)
    results.method_name = f"FedQHD-GD ({steps} steps)"
    print(f"\n{'=' * 60}\nTraining {results.method_name}\n{'=' * 60}")
    start = time.time()

    envs = [create_env(args, client_id=i) for i in range(args.agent_num)]
    state_dim, action_dim = envs[0].state_dim, envs[0].action_dim
    agents = _make_qhd_clients(args, state_dim, action_dim)
    anchors = collect_anchor_states(args, action_dim, args.anchor_set_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = [torch.tensor(_features(a, anchors), dtype=torch.float64, device=device) for a in agents]
    # step size 1/L with L = ||X_i||_2² (the gradient below is scaled by m), computed once
    inv_L = [1.0 / max(float(torch.linalg.matrix_norm(Xi, ord=2)) ** 2, 1e-12) for Xi in X]
    m = len(anchors)
    lam = float(getattr(args, 'ridge_lambda', None) or 1e-4)
    lr = float(getattr(args, 'gd_lr', None) or 1.0)
    K = args.qhd_agg_interval

    rewards, successes, fit_rmse = [], [], []
    for episode in tqdm.tqdm(range(episodes), desc=results.method_name):
        out = [_local_episode(a, e, args) for a, e in zip(agents, envs)]
        rewards.append(float(np.mean([o[0] for o in out])))
        successes.append(float(np.mean([o[1] for o in out])))

        if (episode + 1) % K == 0:
            Ws = [torch.tensor(a.model_vectors.T, dtype=torch.float64, device=device)
                  for a in agents]                                         # (D_i, |A|)
            Q_glob = torch.stack([X[i] @ Ws[i] for i in range(len(agents))]).mean(0)
            round_rmse = []
            for i, a in enumerate(agents):
                W = Ws[i]
                for _ in range(steps):
                    grad = (X[i].T @ (X[i] @ W - Q_glob) + lam * W) / m
                    W = W - lr * m * inv_L[i] * grad
                a.model_vectors = W.T.cpu().numpy()
                round_rmse.append(float(torch.sqrt(torch.mean((X[i] @ W - Q_glob) ** 2))))
            fit_rmse.append(round_rmse)

        for a in agents:
            a.decay_exploration()

    diag = {'gd_steps': steps, 'gd_lr': lr, 'lambda': lam, 'm': m,
            'anchor_fit_rmse': fit_rmse}
    return _finish(results, rewards, successes, start, diag, agents, args)


# ── FedHPD (Jiang et al., 2025): heterogeneous policy distillation ──────────────

# Agents 1–5 of FedHPD Table 4 (CartPole) and Table 5 (LunarLander): (hidden widths,
# activations, Adam learning rate). Acrobot and MountainCar are not covered by the FedHPD
# paper; they use the CartPole configurations (small classic-control tasks).
FEDHPD_AGENTS = {
    'CartPole': [([128], ['relu'], 1e-3), ([32, 32], ['relu'] * 2, 2e-3),
                 ([16, 16, 32], ['tanh'] * 3, 4e-3), ([8, 8, 8], ['relu'] * 3, 5e-4),
                 ([32, 32, 32], ['tanh'] * 3, 3e-3)],
    'LunarLander': [([128, 128, 256], ['relu'] * 3, 5e-4), ([64, 64], ['relu'] * 2, 1e-3),
                    ([128, 128], ['tanh'] * 2, 4e-4), ([128, 256], ['relu'] * 2, 6e-4),
                    ([256, 256], ['tanh'] * 2, 3e-4)],
}
FEDHPD_AGENTS['Acrobot'] = FEDHPD_AGENTS['CartPole']
FEDHPD_AGENTS['MountainCar'] = FEDHPD_AGENTS['CartPole']


class ReinforceAgent:
    """Vanilla REINFORCE with a categorical MLP policy (one update per episode).

    Returns-to-go are standardized within each episode (variance reduction, no critic).
    """

    def __init__(self, state_dim, action_dim, widths, activations, lr, gamma=0.99):
        acts = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh}
        layers, d = [], state_dim
        for w, a in zip(widths, activations):
            layers += [torch.nn.Linear(d, w), acts[a]()]
            d = w
        layers.append(torch.nn.Linear(d, action_dim))
        self.policy = torch.nn.Sequential(*layers)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lr = lr
        self.action_dim = action_dim
        self._logps, self._rewards = [], []

    def probs(self, states) -> torch.Tensor:
        S = torch.as_tensor(np.asarray(states, dtype=np.float32))
        return torch.softmax(self.policy(S), dim=-1)

    def choose_action(self, state) -> int:
        dist = torch.distributions.Categorical(logits=self.policy(
            torch.as_tensor(np.asarray(state, dtype=np.float32))))
        a = dist.sample()
        self._logps.append(dist.log_prob(a))
        return int(a.item())

    def greedy_action(self, state) -> int:
        with torch.no_grad():
            return int(torch.argmax(self.policy(
                torch.as_tensor(np.asarray(state, dtype=np.float32)))).item())

    def finish_episode(self):
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns = torch.tensor(returns[::-1], dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(self._logps) * returns).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._logps, self._rewards = [], []

    def decay_exploration(self):
        pass


def train_fedhpd(episodes: int, args) -> ExperimentResults:
    """FedHPD (Jiang et al., 2025, Algorithm 1) with N heterogeneous REINFORCE agents.

    Local training: one REINFORCE episode per round. Every ``fedhpd_interval`` (d) rounds each
    agent uploads π_k(·|S_p) on the public state set, the server averages them,
    P̄ = (1/K) Σ_k π_k(·|S_p), and each agent takes ``fedhpd_kd_steps`` Adam steps on
    KL(π_k(·|S_p) ‖ P̄) (λ = 1). The public state set is the shared random-rollout anchor set
    S_ref (size --anchor_set_size), i.e. the same states FedQHD uses.
    """
    results = ExperimentResults()
    results.method_name = "FedHPD"
    print(f"\n{'=' * 60}\nTraining {results.method_name}\n{'=' * 60}")
    start = time.time()
    torch.set_num_threads(1)

    envs = [create_env(args, client_id=i) for i in range(args.agent_num)]
    state_dim, action_dim = envs[0].state_dim, envs[0].action_dim
    cfgs = FEDHPD_AGENTS[args.env]
    gamma = float(getattr(args, 'fedhpd_gamma', None) or 0.99)
    agents = [ReinforceAgent(state_dim, action_dim, *cfgs[i % len(cfgs)], gamma=gamma)
              for i in range(args.agent_num)]
    S_p = collect_anchor_states(args, action_dim, args.anchor_set_size)
    d = int(getattr(args, 'fedhpd_interval', None) or 5)
    kd_steps = int(getattr(args, 'fedhpd_kd_steps', None) or 1)

    rewards, successes, kl_hist = [], [], []
    for episode in tqdm.tqdm(range(episodes), desc=results.method_name):
        ep_r, ep_s = [], []
        for agent, env in zip(agents, envs):
            state, _ = env.reset()
            done, total = False, 0.0
            while not done:
                action = agent.choose_action(state)
                state, reward, done, _, _ = env.step(action)
                agent._rewards.append(reward)
                total += reward
            agent.finish_episode()
            ep_r.append(total)
            ep_s.append(1 if env.is_success(state) else 0)
        rewards.append(float(np.mean(ep_r)))
        successes.append(float(np.mean(ep_s)))

        if (episode + 1) % d == 0:
            with torch.no_grad():
                P_bar = torch.stack([a.probs(S_p) for a in agents]).mean(0)
            kls = []
            for a in agents:
                for _ in range(kd_steps):
                    P = a.probs(S_p).clamp_min(1e-8)
                    kl = torch.sum(P * (torch.log(P) - torch.log(P_bar.clamp_min(1e-8))), dim=1).mean()
                    a.opt.zero_grad()
                    kl.backward()
                    a.opt.step()
                kls.append(float(kl.item()))
            kl_hist.append(kls)

    diag = {'interval_d': d, 'kd_steps': kd_steps, 'public_states': len(S_p),
            'agent_configs': [list(map(str, c)) for c in cfgs[:args.agent_num]],
            'kl_to_consensus': kl_hist}
    return _finish(results, rewards, successes, start, diag, agents, args)
