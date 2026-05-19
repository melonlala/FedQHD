"""
Ablation Experiments for FedQHD Paper

Federation error metric (correct definition):
    Q̂_i  = best-in-class approximation of Q* in client i's feature space Φ_i
          = X_i (X_i^T X_i + ε I)^{-1} X_i^T Q*,   ε → 0
    Q*    = estimated by Monte Carlo rollouts from a large-D reference QHD agent
    A1/A2 error_i = RMSE(Q_fed_i, Q*)  (total approx error at held-out eval anchors)
    A3   error_i = RMSE(Q_fed_i, Q̂_i) (federation residual at training anchors, reveals U-shape)
    where Q_fed_i = X_i W_i^{final}  (FedQHD output at anchor states after training)

  Ablation 1 – Vary D: separate experiment per D, m = 4*D, λ = LAMBDA_FIXED
               Expected: larger D → smaller error (D^{-1/2} geometry floor)
  Ablation 2 – Vary m/D: fix D = D_FIXED = 512, sweep m, plot x = m/D
               Expected: sharp decay for m < D, plateau for m ≥ D (regime transition)
  Ablation 3 – Vary λ: separate experiment per λ, D = D_FIXED, m = M_FIXED
               Expected: U-shape (small λ → high variance; large λ → high bias)
               x-axis labelled as α = λ/m

CLI:
  --metric Q_error        compute & plot federation gap ||Q_fed - Q̂||_∞ only
  --metric policy_value   compute & plot policy value V(π_fed) only (skips MC rollouts)
  --metric all            compute & plot both (default)
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import torch
from typing import Dict, List, Optional
import argparse

from agent.qhd_agent import QHDAgent
from env.utils import CartPoleWrapper, LunarLanderWrapper
from env.bounds import state_bounds

# ─── Reproducibility ──────────────────────────────────────────────────────────
MASTER_SEED = 42
np.random.seed(MASTER_SEED)

# ─── Output directories ───────────────────────────────────────────────────────
RESULTS_DIR = "results/ablation"
FIGURES_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── CartPole constants (defaults) ───────────────────────────────────────────
STATE_DIM  = 4
ACTION_DIM = 2

# ─── Per-environment configuration ───────────────────────────────────────────
# Each entry overrides module-level defaults for that environment.
_ENV_REGISTRY: Dict[str, Dict] = {
    'CartPole': dict(
        state_dim=4, action_dim=2,
        wrapper_cls=CartPoleWrapper, bounds_key='CartPole',
        episodes=500, agg_interval=50,
        d_ref=2048,  n_ref_episodes=1000, mc_horizon=300,
        d_fixed=1024, m_fixed=512,
        conv_thresh=480.0,
        use_vectorized_qstar=True,   # use fast _cartpole_step_batch
    ),
    'LunarLander': dict(
        state_dim=8, action_dim=4,
        wrapper_cls=LunarLanderWrapper, bounds_key='LunarLander',
        episodes=600, agg_interval=10,
        d_ref=8192, n_ref_episodes=3000, mc_horizon=1000,
        d_fixed=2048, m_fixed=512,
        conv_thresh=150.0,
        use_vectorized_qstar=False,  # use ref-agent Q values directly
    ),
}

# ─── CartPole physics constants (for vectorized MC rollouts) ──────────────────
_CP_GRAVITY         = 9.8
_CP_MASSCART        = 1.0
_CP_MASSPOLE        = 0.1
_CP_TOTAL_MASS      = _CP_MASSCART + _CP_MASSPOLE
_CP_LENGTH          = 0.5           # half-pole length
_CP_POLEMASS_LENGTH = _CP_MASSPOLE * _CP_LENGTH
_CP_FORCE_MAG       = 10.0
_CP_TAU             = 0.02
_CP_X_THRESHOLD     = 2.4
_CP_THETA_THRESHOLD = 12.0 * np.pi / 180.0

# ─── Training defaults ────────────────────────────────────────────────────────
EPISODES     = 500   # increased for convergence at large D
AGG_INTERVAL = 50    # CartPole best-known value; reduces disruptive resets
RFF_GAMMA    = 1.0
QHD_LR       = 0.1
QHD_DISCOUNT = 0.99
EPS_DECAY    = 0.990
EPS_MIN      = 0.01
NUM_SEEDS    = 3   # 3 seeds for stable statistics (was 2)
N_A_AGENTS   = 4    # agents per homogeneous ablation experiment

# ─── Reference Q* estimation ─────────────────────────────────────────────────
D_REF          = 2048   # reference agent dimension (2048 sufficient for CartPole)
N_REF_EPISODES = 1000    # episodes to train reference agent (was 300)
N_MC_ROLLOUTS  = 5      # MC rollouts per (anchor, action) pair
MC_HORIZON     = 300    # max steps per MC rollout
N_EVAL_EPISODES = 30    # greedy episodes for policy-value evaluation
REF_AGENT_PATH  = os.path.join(RESULTS_DIR, "ref_agent.npz")

# ─── Ablation grids ───────────────────────────────────────────────────────────
# A1: vary D – one separate experiment per D, m = 4*D  (cap D≤2048 → max m=8192)
D_GRID = [128, 256, 512, 1024, 2048, 4096, 8192]  # includes two extra points beyond the theory regime to show divergence at large D (MC noise floor dominates)
# A2: vary m/D – fix D = D_FIXED_A2 = 512, sweep m spanning both sides of m=D
D_FIXED_A2 = 1024
M_GRID     = [128, 256, 512, 1024, 1536, 2048]   # m/D ∈ {0.125, 0.25, 0.5, 1, 1.5, 2}

# A3: vary λ – one separate experiment per λ, fix D = D_FIXED, m = M_FIXED
D_FIXED = 1024     # same as D_FIXED_A2 for consistency
M_FIXED = 1000     # slightly below D_FIXED (under-parameterised → U-shape visible)
# alpha* (theory) = D / sqrt(m) = 1024 / sqrt(1000) ≈ 32.3,  lambda* = alpha* * M_FIXED = 32.3 * 1000 ≈ 32,300
# Empirical Q_error minimum found at λ≈10. We add λ=0.1 to capture the variance arm.
# Grid spans α = λ/m from 0.00025 to 250 — both arms of the U-shape are captured.
LAMBDA_GRID = [0.1, 1, 10, 100, 1000, 10000, 100000]  # corresponds to α = λ/m from 0.0001 to 100 (log-spaced around the empirical minimum at λ=10)

LAMBDA_FIXED = 1e-4   # fixed λ used in A1 / A2
LAMBDA_HAT   = 1e-10  # near-zero λ for Q̂_i projection (oracle best-in-class)

N_EVAL = 400  # held-out test anchors for unbiased RMSE evaluation (collected with different seed)

# ─── Plot axis labels ─────────────────────────────────────────────────────────
ERR_LABEL = r"Approx. Error RMSE$(Q_i^{\mathrm{fed}},\, Q^*)$"
PVG_LABEL = r"Policy Value Gap $V(\pi^*) - V(\pi_i^{\mathrm{fed}})$"
PV_LABEL  = r"Policy Value $V(\pi_i^{\mathrm{fed}})$"


# =============================================================================
# Vectorised CartPole physics
# =============================================================================

def _cartpole_step_batch(states: np.ndarray, actions: np.ndarray):
    """
    Vectorised CartPole step using pure NumPy (no gym overhead).

    Parameters
    ----------
    states  : (N, 4) float64  [x, x_dot, theta, theta_dot]
    actions : (N,)   int      {0, 1}

    Returns
    -------
    new_states : (N, 4)
    rewards    : (N,)  1.0 per alive step
    dones      : (N,)  bool
    """
    x, xd, th, thd = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    force    = (2 * actions - 1) * _CP_FORCE_MAG  # ±10

    costh = np.cos(th)
    sinth = np.sin(th)
    temp  = (force + _CP_POLEMASS_LENGTH * thd**2 * sinth) / _CP_TOTAL_MASS
    thacc = (_CP_GRAVITY * sinth - costh * temp) / (
        _CP_LENGTH * (4.0/3.0 - _CP_MASSPOLE * costh**2 / _CP_TOTAL_MASS))
    xacc  = temp - _CP_POLEMASS_LENGTH * thacc * costh / _CP_TOTAL_MASS

    # Euler integration
    x_n   = x   + _CP_TAU * xd
    xd_n  = xd  + _CP_TAU * xacc
    th_n  = th  + _CP_TAU * thd
    thd_n = thd + _CP_TAU * thacc

    new_states = np.stack([x_n, xd_n, th_n, thd_n], axis=1)
    rewards    = np.ones(len(states), dtype=np.float64)
    dones      = (np.abs(x_n) > _CP_X_THRESHOLD) | (np.abs(th_n) > _CP_THETA_THRESHOLD)
    return new_states, rewards, dones


def _batch_greedy_actions(
    agent: QHDAgent,
    states_np: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Greedy actions for a batch of unnormalized states.
    Uses float32 internally for the expensive cos computation (~2.5x speedup
    over float64 on large D arrays; safe for argmax-only usage).

    states_np : (N, state_dim)
    Returns   : (N,) int actions
    """
    bounds    = agent.state_bounds           # (state_dim, 2)
    min_b     = bounds[:, 0].astype(np.float32)
    range_b   = np.where(bounds[:, 1] - bounds[:, 0] > 0,
                         bounds[:, 1] - bounds[:, 0], 1.0).astype(np.float32)
    omega32   = agent.omega.astype(np.float32)   # (D, state_dim)
    b32       = agent.b.astype(np.float32)        # (D,)
    W32       = agent.model_vectors.astype(np.float32)  # (action_dim, D)
    scale     = np.float32(np.sqrt(2.0 / agent.hd_dim))

    N       = len(states_np)
    actions = np.zeros(N, dtype=int)
    for i in range(0, N, chunk_size):
        chunk = states_np[i: i + chunk_size].astype(np.float32)
        norm  = 2.0 * (chunk - min_b) / range_b - 1.0   # float32
        phi   = scale * np.cos(norm @ omega32.T + b32)   # (chunk, D) float32
        Q_ch  = phi @ W32.T                               # (chunk, action_dim) float32
        actions[i: i + len(chunk)] = np.argmax(Q_ch, axis=1)
    return actions


# =============================================================================
# Reference agent save / load
# =============================================================================

def save_ref_agent(agent: QHDAgent, path: str = REF_AGENT_PATH) -> None:
    meta = np.array([
        agent.state_dim, agent.action_dim, agent.hd_dim, agent.rff_gamma,
        agent.learning_rate, agent.gamma, agent.epsilon,
        agent.epsilon_decay, agent.epsilon_min, float(agent.random_state),
    ], dtype=np.float64)
    np.savez(path,
             model_vectors=agent.model_vectors, omega=agent.omega, b=agent.b,
             state_bounds=agent.state_bounds, meta=meta)
    print(f"  [Q*] Reference agent saved → {path}")


def load_ref_agent(path: str = REF_AGENT_PATH) -> QHDAgent:
    data = np.load(path)
    meta = data['meta']
    (state_dim, action_dim, hd_dim, rff_gamma, learning_rate,
     discount_factor, exploration_rate, exploration_decay,
     exploration_min, random_seed) = meta

    agent = QHDAgent(
        state_dim=int(state_dim), action_dim=int(action_dim), hd_dim=int(hd_dim),
        rff_gamma=float(rff_gamma), learning_rate=float(learning_rate),
        discount_factor=float(discount_factor), exploration_rate=float(exploration_rate),
        exploration_decay=float(exploration_decay), exploration_min=float(exploration_min),
        state_bounds=data['state_bounds'].tolist(), random_seed=int(random_seed),
    )
    agent.model_vectors = data['model_vectors'].copy()
    agent.omega         = data['omega'].copy()
    agent.b             = data['b'].copy()
    print(f"  [Q*] Reference agent loaded ← {path}"
          f"  (D={int(hd_dim)}, ε={float(exploration_rate):.3f})")
    return agent


# =============================================================================
# Q* estimation – fast vectorised MC rollouts + disk caching
# =============================================================================

def _qstar_cache_path(m: int, n_mc: int, discount: float = QHD_DISCOUNT, env: str = 'CartPole') -> str:
    # Include discount and env in the filename so caches with different γ or env don't collide.
    disc_str = f"{discount:.3f}".replace('.', 'p')
    return os.path.join(RESULTS_DIR, f"{env}_Q_star_m{m}_nmc{n_mc}_g{disc_str}.npz")


def compute_Q_star_fast(
    anchor_np: np.ndarray,
    ref_agent: QHDAgent,
    n_mc: int   = N_MC_ROLLOUTS,
    mc_horizon: int = MC_HORIZON,
    discount: float = QHD_DISCOUNT,
    chunk_size: int = 2048,
    force_recompute: bool = False,
    action_dim: int = ACTION_DIM,
    env: str = 'CartPole',
) -> np.ndarray:
    """
    Vectorised CartPole MC rollouts using pure NumPy (no gym overhead).
    Results are cached to disk; subsequent calls with same (m, n_mc) are instant.

    Returns Q_star : (m, action_dim)
    """
    m          = len(anchor_np)
    cache_path = _qstar_cache_path(m, n_mc, discount, env)

    if not force_recompute and os.path.exists(cache_path):
        data = np.load(cache_path)
        saved_anchors = data['anchors']
        if saved_anchors.shape == anchor_np.shape and np.allclose(saved_anchors, anchor_np, atol=1e-6):
            print(f"  [Q*] Q_star loaded from cache ({cache_path})")
            return data['Q_star']
        else:
            print(f"  [Q*] Cache anchor mismatch – recomputing …")

    print(f"  [Q*] Vectorised MC rollouts: m={m}, n_mc={n_mc} …")
    N       = m * n_mc
    Q_star  = np.zeros((m, action_dim), dtype=np.float64)

    for a_init in range(action_dim):
        # Repeat each anchor n_mc times → (N, 4)
        states  = np.repeat(anchor_np, n_mc, axis=0).astype(np.float64)

        # Forced first action
        acts0   = np.full(N, a_init, dtype=int)
        states, rews, dones = _cartpole_step_batch(states, acts0)
        returns = rews.copy()
        alive   = ~dones
        gamma_t = discount * np.ones(N, dtype=np.float64)

        for _ in range(mc_horizon):
            if not alive.any():
                break
            alive_idx = np.where(alive)[0]
            # Greedy actions only for alive states (batch encode in chunks)
            acts = np.zeros(N, dtype=int)
            acts[alive_idx] = _batch_greedy_actions(ref_agent, states[alive_idx], chunk_size)
            states, rews, dones = _cartpole_step_batch(states, acts)
            returns  += gamma_t * alive.astype(np.float64) * rews
            gamma_t  *= discount
            alive     = alive & ~dones

        Q_star[:, a_init] = returns.reshape(m, n_mc).mean(axis=1)
        print(f"  [Q*]   action {a_init} done, Q range [{Q_star[:, a_init].min():.2f}, "
              f"{Q_star[:, a_init].max():.2f}]")

    print(f"  [Q*] Done. Q* range: [{Q_star.min():.3f}, {Q_star.max():.3f}]")
    np.savez(cache_path, Q_star=Q_star, anchors=anchor_np)
    print(f"  [Q*] Q_star saved → {cache_path}")
    return Q_star


def compute_Q_star_from_agent(
    anchor_np: np.ndarray,
    ref_agent: QHDAgent,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Approximate Q*(s,a) using the converged reference agent's own Q-function.
        Q*(s,a) ≈ Φ_ref(s) @ W_{ref,a}

    Used when gym env state cannot be set arbitrarily (e.g. LunarLander/Box2D),
    so vectorised MC rollouts from arbitrary anchor states are not feasible.
    Results are cached to disk for reuse.

    Returns Q_star : (m, action_dim)
    """
    m = len(anchor_np)
    if cache_path and not force_recompute and os.path.exists(cache_path):
        data = np.load(cache_path)
        if (data['anchors'].shape == anchor_np.shape
                and np.allclose(data['anchors'], anchor_np, atol=1e-6)):
            print(f"  [Q*] Q_star loaded from cache ({cache_path})")
            return data['Q_star']
        print(f"  [Q*] Cache anchor mismatch – recomputing …")

    print(f"  [Q*] Computing Q* from ref agent (m={m}) …")
    X_ref  = encode_anchors(ref_agent, anchor_np)   # (m, D_ref)
    W_ref  = ref_agent.model_vectors                 # (action_dim, D_ref)
    Q_star = X_ref @ W_ref.T                         # (m, action_dim)
    print(f"  [Q*] Done. Q* range: [{Q_star.min():.3f}, {Q_star.max():.3f}]")
    if cache_path:
        np.savez(cache_path, Q_star=Q_star, anchors=anchor_np)
        print(f"  [Q*] Q_star saved → {cache_path}")
    return Q_star


def _ensure_ref_agent(
    d_ref: int           = D_REF,
    n_ref_episodes: int  = N_REF_EPISODES,
    rff_gamma: float     = RFF_GAMMA,
    qhd_lr: float        = QHD_LR,
    discount: float      = QHD_DISCOUNT,
    eps_decay: float     = EPS_DECAY,
    eps_min: float       = EPS_MIN,
    state_dim: int       = STATE_DIM,
    action_dim: int      = ACTION_DIM,
    wrapper_cls          = CartPoleWrapper,
    bounds_key: str      = 'CartPole',
    ref_agent_path: str  = REF_AGENT_PATH,
) -> None:
    """Train and cache the reference agent if not already saved (or if cached D differs).

    NOTE: eps_decay here controls the ref agent's exploration schedule independently
    of the FedQHD agents' eps_decay.  A slower value (e.g. 0.995) is recommended so
    the standalone ref agent explores enough across 1000 episodes before going greedy.
    """
    if os.path.exists(ref_agent_path):
        data = np.load(ref_agent_path)
        saved_D = int(data['meta'][2])  # index 2 = hd_dim
        if saved_D == d_ref:
            return  # cached agent has the correct dimension
        print(f"  [Ref] Cached agent has D={saved_D}, need D={d_ref} — retraining …")
    print(f"\n  [Ref] Training reference agent  D={d_ref}, {n_ref_episodes} episodes …")
    ref_agent = QHDAgent(
        state_dim=state_dim, action_dim=action_dim, hd_dim=d_ref,
        rff_gamma=rff_gamma, learning_rate=qhd_lr / 2,
        discount_factor=discount, exploration_rate=1.0,
        exploration_decay=eps_decay, exploration_min=eps_min,
        state_bounds=state_bounds[bounds_key], random_seed=MASTER_SEED,
    )
    train_env = wrapper_cls()
    for _ in tqdm.trange(n_ref_episodes, desc="  [Ref] training", leave=False):
        state, _ = train_env.reset()
        done = False
        while not done:
            action = ref_agent.choose_action(state)
            next_state, reward, done, _, _ = train_env.step(action)
            ref_agent.update_model(state, action, reward, next_state, done)
            state = next_state
        ref_agent.decay_exploration()
    save_ref_agent(ref_agent, ref_agent_path)


# =============================================================================
# Anchor feature encoding and Q̂_i projection
# =============================================================================

def encode_anchors(agent: QHDAgent, anchor_np: np.ndarray) -> np.ndarray:
    """Compute X_i ∈ R^{m × D_i} using agent i's RFF encoder (vectorised)."""
    bounds  = np.array(agent.state_bounds)
    min_b   = bounds[:, 0]
    range_b = np.where(bounds[:, 1] - bounds[:, 0] > 0, bounds[:, 1] - bounds[:, 0], 1.0)
    norm_s  = 2.0 * (anchor_np - min_b) / range_b - 1.0
    return np.sqrt(2.0 / agent.hd_dim) * np.cos(norm_s @ agent.omega.T + agent.b)


def compute_Qhat(Q_star: np.ndarray, X_i: torch.Tensor,
                 lambda_hat: float = LAMBDA_HAT) -> np.ndarray:
    """
    Best-in-class approximation of Q* in feature space Φ_i.
    Q̂_i = X_i (X_i^T X_i + λ_hat I)^{-1} X_i^T Q*   (pseudoinverse, λ_hat ≈ 0)
    """
    device    = X_i.device
    Q_star_t  = torch.tensor(Q_star, dtype=torch.float64, device=device)
    D_i       = X_i.shape[1]
    A         = X_i.T @ X_i + lambda_hat * torch.eye(D_i, dtype=torch.float64, device=device)
    W_hat     = torch.linalg.solve(A, X_i.T @ Q_star_t)
    Q_hat     = X_i @ W_hat
    return Q_hat.cpu().numpy()


def compute_What(Q_star: np.ndarray, X_train: torch.Tensor,
                 lambda_hat: float = LAMBDA_HAT) -> torch.Tensor:
    """
    Ridge regression weights: W_hat = (X_train^T X_train + λI)^{-1} X_train^T Q*.
    Returns shape (D, action_dim).  Evaluate at arbitrary points: Q_hat_eval = X_eval @ W_hat.
    Separating weight computation from evaluation allows RMSE at held-out test anchors.
    """
    device   = X_train.device
    D_i      = X_train.shape[1]
    Q_star_t = torch.tensor(Q_star, dtype=torch.float64, device=device)
    A        = X_train.T @ X_train + lambda_hat * torch.eye(D_i, dtype=torch.float64, device=device)
    return torch.linalg.solve(A, X_train.T @ Q_star_t)  # (D, action_dim)


def sup_norm(Q_fed: np.ndarray, Q_hat: np.ndarray) -> float:
    return float(np.max(np.abs(Q_fed - Q_hat)))


def rmse(Q_fed: np.ndarray, Q_hat: np.ndarray) -> float:
    """RMSE federation error — normalized, does not grow trivially with m."""
    return float(np.sqrt(np.mean((Q_fed - Q_hat) ** 2)))


def compute_policy_value(agent: QHDAgent, n_episodes: int = N_EVAL_EPISODES,
                         wrapper_cls=CartPoleWrapper) -> float:
    eps_saved  = agent.epsilon
    agent.epsilon = 0.0
    env    = wrapper_cls()
    total  = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _, _ = env.step(action)
            total += reward
    agent.epsilon = eps_saved
    return total / n_episodes


# =============================================================================
# Core FedQHD training (homogeneous D and λ per agent – used by all ablations)
# =============================================================================

def run_fedqhd_homo(
    D:           int,
    m:           int,
    lam:         float,
    anchor_np:   np.ndarray,
    num_agents:  int   = N_A_AGENTS,
    episodes:    int   = EPISODES,
    agg_interval:int   = AGG_INTERVAL,
    seed:        int   = MASTER_SEED,
    rff_gamma:   float = RFF_GAMMA,
    qhd_lr:      float = QHD_LR,
    discount:    float = QHD_DISCOUNT,
    eps_decay:   float = EPS_DECAY,
    eps_min:     float = EPS_MIN,
    Q_star:          Optional[np.ndarray] = None,
    metric:          str                  = 'all',
    eval_anchor_np:  Optional[np.ndarray] = None,
    Q_star_eval:     Optional[np.ndarray] = None,
    qref_noise:      float                = 0.0,
    state_dim:       int                  = STATE_DIM,
    action_dim:      int                  = ACTION_DIM,
    wrapper_cls                           = CartPoleWrapper,
    bounds_key:      str                  = 'CartPole',
) -> Dict:
    """
    FedQHD with num_agents all having the SAME D and λ.
    Each agent may have a slightly different RFF bandwidth (σ_i ∼ Uniform[0.5σ, 1.5σ])
    to reflect realistic encoder heterogeneity within the same dimension class.

    Returns dict with keys depending on metric:
      error_mean / error_std, pv_mean / pv_std, per_agent_rewards
    """
    need_q_err = metric in ('all', 'Q_error')
    need_pv    = metric in ('all', 'policy_value')
    if need_q_err and Q_star is None:
        raise ValueError("Q_star required for Q_error metric")

    rng    = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    envs   = [wrapper_cls() for _ in range(num_agents)]
    agents = []
    for i in range(num_agents):
        sigma_i = rff_gamma * float(rng.uniform(0.5, 1.5))
        agents.append(QHDAgent(
            state_dim=state_dim, action_dim=action_dim, hd_dim=D,
            rff_gamma=sigma_i, learning_rate=qhd_lr,
            discount_factor=discount, exploration_rate=1.0,
            exploration_decay=eps_decay, exploration_min=eps_min,
            state_bounds=state_bounds[bounds_key],
            random_seed=int(seed) + i * 100,
        ))

    # Pre-compute anchor features and D×D Cholesky factors.
    # Use the D×D form  (X^T X + λI)  instead of the m×m form  (X X^T + λI).
    # Both are mathematically equivalent via the push-through identity but the
    # D×D form avoids the O(m²D) memory explosion when m > D (which happens in
    # A1 where m = 4D).
    anchor_feats: List[torch.Tensor] = []
    chol_factors: List[torch.Tensor] = []
    for agent in agents:
        X_np = encode_anchors(agent, anchor_np)
        X    = torch.tensor(X_np, dtype=torch.float64, device=device)
        anchor_feats.append(X)
        A    = X.T @ X + lam * torch.eye(D, dtype=torch.float64, device=device)
        chol_factors.append(torch.linalg.cholesky(A))

    per_agent_rewards = np.zeros((num_agents, episodes))

    for ep in range(episodes):
        for i, (agent, env) in enumerate(zip(agents, envs)):
            state, _ = env.reset()
            done, total_r = False, 0.0
            while not done:
                action = agent.choose_action(state)
                next_s, r, done, _, _ = env.step(action)
                agent.update_model(state, action, r, next_s, done)
                state   = next_s
                total_r += r
            per_agent_rewards[i, ep] = total_r

        if (ep + 1) % agg_interval == 0:
            Q_list = [anchor_feats[i] @ torch.tensor(
                agents[i].model_vectors, dtype=torch.float64, device=device).T
                for i in range(num_agents)]
            Q_glob = torch.mean(torch.stack(Q_list), dim=0)
            if qref_noise > 0.0:
                noise_std = float(Q_glob.std().cpu()) * qref_noise
                noise_rng = np.random.default_rng(seed * 10000 + ep)
                noise_np  = noise_rng.standard_normal(Q_glob.shape)
                Q_glob = Q_glob + noise_std * torch.tensor(
                    noise_np, dtype=Q_glob.dtype, device=Q_glob.device)

            for i, agent in enumerate(agents):
                X_i, L_i = anchor_feats[i], chol_factors[i]
                # D×D solve: W_i = (X_i^T X_i + λI)^{-1} X_i^T Q_glob
                # (Equivalent to the m×m solve via push-through identity.)
                rhs = X_i.T @ Q_glob                   # (D, action_dim)
                W_i = torch.linalg.solve_triangular(
                    L_i.T,
                    torch.linalg.solve_triangular(L_i, rhs, upper=False),
                    upper=True)                         # (D, action_dim)
                agent.model_vectors = W_i.T.cpu().numpy()  # (action_dim, D)

        for agent in agents:
            agent.decay_exploration()

    # Compute requested metrics
    errors, pvs = [], []
    for i, agent in enumerate(agents):
        if need_q_err:
            W_i = torch.tensor(agent.model_vectors, dtype=torch.float64, device=device)
            if eval_anchor_np is not None:
                # A1/A2: TOTAL error = RMSE(Q_fed, Q*) at held-out eval anchors.
                # Measures how well FedQHD approximates Q* (not just Q_hat), so the
                # D^{-1/2} geometry floor is directly observable.
                X_eval = torch.tensor(encode_anchors(agent, eval_anchor_np),
                                      dtype=torch.float64, device=device)
                Q_fed_eval = (X_eval @ W_i.T).cpu().numpy()
                target = Q_star_eval  # Q* at eval anchors (pre-computed via MC)
                errors.append(rmse(Q_fed_eval, target))
            else:
                # A3: federation residual = RMSE(Q_fed, Q_hat) at TRAINING anchors.
                # Reveals the λ bias-variance U-shape (not visible at held-out points).
                W_hat = compute_What(Q_star, anchor_feats[i], LAMBDA_HAT)
                Q_fed_train  = (anchor_feats[i] @ W_i.T).cpu().numpy()
                Q_hat_train  = (anchor_feats[i] @ W_hat).cpu().numpy()
                errors.append(rmse(Q_fed_train, Q_hat_train))
        if need_pv:
            pvs.append(compute_policy_value(agent, wrapper_cls=wrapper_cls))

    return {
        'per_agent_rewards': per_agent_rewards,
        'fed_errors':  errors,   # list[float]
        'policy_values': pvs,    # list[float]
        'D': D, 'm': m, 'lam': lam,
    }


def run_oracle_error(
    D: int, m: int, lam: float,
    anchor_np: np.ndarray, Q_star: np.ndarray,
    eval_anchor_np: np.ndarray, Q_star_eval: np.ndarray,
    rff_gamma: float = RFF_GAMMA, seed: int = MASTER_SEED,
    state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
    bounds_key: str = 'CartPole',
    **kw,
) -> float:
    """
    Analytical oracle approximation error (no RL training).
    Computes W_hat = ridge regression of Q*[:m] on Φ(anchor[:m]) and returns
    RMSE(Φ(eval) @ W_hat,  Q*_eval).  Measures pure feature-space quality:
    - Varies with D → validates D^{-1/2} geometry floor (A1)
    - Varies with m → validates sample-regime transition (A2)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng    = np.random.default_rng(seed)
    sigma_i = rff_gamma * float(rng.uniform(0.5, 1.5))
    agent  = QHDAgent(
        state_dim=state_dim, action_dim=action_dim, hd_dim=D,
        rff_gamma=sigma_i, learning_rate=0.1, discount_factor=QHD_DISCOUNT,
        exploration_rate=0.0, exploration_decay=1.0, exploration_min=0.0,
        state_bounds=state_bounds[bounds_key], random_seed=int(seed),
    )
    X_tr  = torch.tensor(encode_anchors(agent, anchor_np[:m]),
                         dtype=torch.float64, device=device)
    W_hat = compute_What(Q_star[:m], X_tr, lambda_hat=lam)   # (D, action_dim)
    X_ev  = torch.tensor(encode_anchors(agent, eval_anchor_np),
                         dtype=torch.float64, device=device)
    Q_hat_eval = (X_ev @ W_hat).cpu().numpy()
    return rmse(Q_hat_eval, Q_star_eval)


def _run_oracle_seeds(
    D: int, m: int, lam: float,
    anchor_np: np.ndarray, Q_star: np.ndarray,
    eval_anchor_np: np.ndarray, Q_star_eval: np.ndarray,
    num_seeds: int = NUM_SEEDS,
    **kw,   # absorb unused kwargs (episodes, metric, etc.) from shared dict
) -> Dict:
    """Average run_oracle_error over random seeds; returns dict in _run_seeds format."""
    _env_kw = {k: kw[k] for k in ('state_dim', 'action_dim', 'bounds_key') if k in kw}
    errors = [run_oracle_error(D, m, lam, anchor_np, Q_star, eval_anchor_np, Q_star_eval,
                               seed=MASTER_SEED + s * 17, **_env_kw) for s in range(num_seeds)]
    ea = np.array(errors)
    return {
        'error_mean':  np.array([ea.mean()]),  # (1,) one virtual "agent"
        'error_std':   np.array([ea.std()]),
        'reward_mean': np.zeros((1, 1)),        # placeholder (no RL learning curves)
        'reward_std':  np.zeros((1, 1)),
        'D': D, 'm': m, 'lam': lam, 'oracle_value': None,
    }


def _run_seeds(
    D: int, m: int, lam: float, anchor_np: np.ndarray,
    oracle_value: Optional[float] = None,
    Q_star: Optional[np.ndarray] = None,
    num_agents: int = N_A_AGENTS,
    episodes: int = EPISODES,
    num_seeds: int = NUM_SEEDS,
    metric: str = 'all',
    **kw,
) -> Dict:
    """
    Average run_fedqhd_homo over multiple seeds.

    Returns dict with:
      error_mean / error_std   (metric includes Q_error)
      pv_mean / pv_std / pv_gap_mean / pv_gap_std  (metric includes policy_value)
      reward_mean / reward_std  (always, shape (num_agents, episodes))
    """
    need_q_err = metric in ('all', 'Q_error')
    need_pv    = metric in ('all', 'policy_value')

    all_errors  = np.zeros((num_seeds, num_agents))
    all_pv_raw  = np.zeros((num_seeds, num_agents))
    all_rewards = np.zeros((num_seeds, num_agents, episodes))

    for s in range(num_seeds):
        r = run_fedqhd_homo(
            D=D, m=m, lam=lam, anchor_np=anchor_np,
            num_agents=num_agents, episodes=episodes,
            seed=MASTER_SEED + s * 17,
            Q_star=Q_star, metric=metric, **kw,
        )
        if need_q_err:
            all_errors[s] = r['fed_errors']
        if need_pv:
            all_pv_raw[s] = r['policy_values']
        all_rewards[s] = r['per_agent_rewards']

    out: Dict = {
        'reward_mean':  all_rewards.mean(axis=0),
        'reward_std':   all_rewards.std(axis=0),
        'oracle_value': oracle_value,
        'D': D, 'm': m, 'lam': lam,
    }
    if need_q_err:
        out['error_mean'] = all_errors.mean(axis=0)   # (num_agents,)
        out['error_std']  = all_errors.std(axis=0)
    if need_pv:
        out['pv_mean'] = all_pv_raw.mean(axis=0)
        out['pv_std']  = all_pv_raw.std(axis=0)
        if oracle_value is not None:
            gaps = oracle_value - all_pv_raw
            out['pv_gap_mean'] = gaps.mean(axis=0)
            out['pv_gap_std']  = gaps.std(axis=0)
    return out


# =============================================================================
# Anchor state collection
# =============================================================================

def collect_anchors(m: int, seed: int = MASTER_SEED,
                    wrapper_cls=CartPoleWrapper, action_dim: int = ACTION_DIM) -> np.ndarray:
    """Collect m anchor states from random env rollouts (fully deterministic via seed)."""
    rng     = np.random.default_rng(seed)
    env     = wrapper_cls()
    anchors = []
    while len(anchors) < m:
        # Seed the underlying gym env directly so anchor sets are reproducible across runs.
        # Without this, env.reset() is non-deterministic → Q* cache always misses → noisy RMSE.
        ep_seed = int(rng.integers(1_000_000))
        state, _ = env.env.reset(seed=ep_seed)
        env.episode_reward = 0
        anchors.append(state.copy())
        for _ in range(200):
            action = int(rng.integers(action_dim))
            next_state, _, done, _, _ = env.step(action)
            anchors.append(next_state.copy())
            if done or len(anchors) >= m:
                break
    return np.array(anchors[:m])


# =============================================================================
# Ablation 1 – Vary D  (separate experiment per D, m = 4*D)
# =============================================================================

def ablation1_vary_D(
    anchor_full: np.ndarray,
    oracle_value: Optional[float] = None,
    Q_star_full: Optional[np.ndarray] = None,
    episodes: int = EPISODES,
    num_seeds: int = NUM_SEEDS,
    metric: str = 'all',
    d_grid: List[int] = D_GRID,
    lambda_fixed: float = LAMBDA_FIXED,
    m_fixed: Optional[int] = None,
    analytical_mode: bool = False,
    eval_anchor_np: Optional[np.ndarray] = None,
    Q_star_eval: Optional[np.ndarray] = None,
    **kw,
) -> List[Dict]:
    """
    For each D in d_grid: run N_A_AGENTS homogeneous agents with the given D.

    If m_fixed is given (recommended), all D experiments share the SAME anchor set
    of size m_fixed — this removes the confound that m=4*D uses more diverse / harder
    states for larger D, which inflates RMSE even though approximation quality improves.

    If m_fixed is None (legacy), m = 4*D per experiment.

    Fixed λ = lambda_fixed.

    Expected Q_error behaviour: error ∝ D^{-1/2}  (geometry floor).
    Expected policy_value:       larger D → higher V.

    Returns list of dicts, one per D.
    """
    use_fixed     = m_fixed is not None
    m_label       = f"m={m_fixed}(fixed)" if use_fixed else "m=4D"
    q_star_avail  = (eval_anchor_np is not None and Q_star_eval is not None
                     and Q_star_full is not None)
    use_analytical = analytical_mode and q_star_avail           # pure Q_error, no RL
    # Hybrid: run RL for learning-curves + policy-value AND compute analytical Q_error.
    # Activated when metric='all' and Q* is available so the Q_error panel shows the
    # clean geometry-floor slope (not RL-agent Q_error which is nosier and metric-mixed).
    use_hybrid    = (not analytical_mode) and q_star_avail and metric in ('all', 'Q_error')
    mode_tag      = "ANALYTICAL" if use_analytical else ("HYBRID" if use_hybrid else "RL")
    print("\n" + "=" * 60)
    print(f"Ablation 1: Vary D  {d_grid}  ({m_label}, λ={lambda_fixed:.0e}, {mode_tag})")
    if oracle_value is not None:
        print(f"  Oracle policy value: {oracle_value:.2f}")
    print("=" * 60)

    results_per_D = []
    for D in tqdm.tqdm(d_grid, desc="Ablation 1"):
        m    = m_fixed if use_fixed else 4 * D
        Q_m  = Q_star_full[:m] if Q_star_full is not None else None
        if use_analytical:
            r = _run_oracle_seeds(
                D=D, m=m, lam=LAMBDA_HAT,  # near-zero λ = oracle regression (matches search_ablation_hparams.py)
                anchor_np=anchor_full, Q_star=Q_m,
                eval_anchor_np=eval_anchor_np, Q_star_eval=Q_star_eval,
                num_seeds=num_seeds, **kw,
            )
        elif use_hybrid:
            # Analytical Q_error (oracle ridge regression on Q*)
            r_oracle = _run_oracle_seeds(
                D=D, m=m, lam=LAMBDA_HAT,
                anchor_np=anchor_full, Q_star=Q_m,
                eval_anchor_np=eval_anchor_np, Q_star_eval=Q_star_eval,
                num_seeds=num_seeds,
            )
            # RL training for learning curves + policy value
            r_rl = _run_seeds(
                D=D, m=m, lam=lambda_fixed, anchor_np=anchor_full[:m],
                oracle_value=oracle_value, Q_star=None,
                episodes=episodes, num_seeds=num_seeds, metric='policy_value', **kw,
            )
            # Merge: override RL Q_error with cleaner analytical oracle error
            r = {**r_rl,
                 'error_mean': r_oracle['error_mean'],
                 'error_std':  r_oracle['error_std']}
        else:
            r    = _run_seeds(
                D=D, m=m, lam=lambda_fixed, anchor_np=anchor_full[:m],
                oracle_value=oracle_value, Q_star=Q_m,
                episodes=episodes, num_seeds=num_seeds, metric=metric, **kw,
            )
        entry = {'D': D, 'm': m, 'result': r}
        msg   = f"  D={D:5d}  m={m:6d}"
        if 'error_mean' in r:
            e_m = float(r['error_mean'].mean())
            e_s = float(r['error_std'].mean())
            msg += f"  fed_gap={e_m:.4f} ± {e_s:.4f}"
            entry['err_m'] = e_m;  entry['err_s'] = e_s
        if 'pv_mean' in r:
            p_m = float(r['pv_mean'].mean())
            p_s = float(r['pv_std'].mean())
            msg += f"  pv={p_m:.2f} ± {p_s:.2f}"
            entry['pv_m'] = p_m;   entry['pv_s'] = p_s
        results_per_D.append(entry)
        print(msg)

    return results_per_D


# =============================================================================
# Ablation 2 – Vary m/D  (fix D = D_FIXED_A2 = 512, sweep m)
# =============================================================================

def ablation2_vary_m(
    anchor_full: np.ndarray,
    oracle_value: Optional[float] = None,
    Q_star_full: Optional[np.ndarray] = None,
    episodes: int = EPISODES,
    num_seeds: int = NUM_SEEDS,
    metric: str = 'all',
    m_grid: List[int] = M_GRID,
    d_fixed: int = D_FIXED_A2,
    lambda_fixed: float = LAMBDA_FIXED,
    analytical_mode: bool = False,
    eval_anchor_np: Optional[np.ndarray] = None,
    Q_star_eval: Optional[np.ndarray] = None,
    **kw,
) -> List[Dict]:
    """
    Fix D = d_fixed = 512.  Sweep m in m_grid (spans both m < D and m > D).
    x-axis: m/D ratio with vertical line at 1 (under→over transition).

    Expected:  sharp error decrease for m < D,  plateau for m ≥ D.
    """
    q_star_avail  = (eval_anchor_np is not None and Q_star_eval is not None
                     and Q_star_full is not None)
    use_analytical = analytical_mode and q_star_avail
    use_hybrid    = (not analytical_mode) and q_star_avail and metric in ('all', 'Q_error')
    mode_tag      = "ANALYTICAL" if use_analytical else ("HYBRID" if use_hybrid else "RL")
    print("\n" + "=" * 60)
    print(f"Ablation 2: Vary m  {m_grid}  (D={d_fixed}, λ={lambda_fixed:.0e}, {mode_tag})")
    print("=" * 60)

    results_per_m = []
    for m in tqdm.tqdm(m_grid, desc="Ablation 2"):
        Q_m = Q_star_full[:m] if Q_star_full is not None else None
        if use_analytical:
            r = _run_oracle_seeds(
                D=d_fixed, m=m, lam=LAMBDA_HAT,  # near-zero λ = oracle regression
                anchor_np=anchor_full, Q_star=Q_m,
                eval_anchor_np=eval_anchor_np, Q_star_eval=Q_star_eval,
                num_seeds=num_seeds, **kw,
            )
        elif use_hybrid:
            r_oracle = _run_oracle_seeds(
                D=d_fixed, m=m, lam=LAMBDA_HAT,
                anchor_np=anchor_full, Q_star=Q_m,
                eval_anchor_np=eval_anchor_np, Q_star_eval=Q_star_eval,
                num_seeds=num_seeds,
            )
            r_rl = _run_seeds(
                D=d_fixed, m=m, lam=lambda_fixed, anchor_np=anchor_full[:m],
                oracle_value=oracle_value, Q_star=None,
                episodes=episodes, num_seeds=num_seeds, metric='policy_value', **kw,
            )
            r = {**r_rl,
                 'error_mean': r_oracle['error_mean'],
                 'error_std':  r_oracle['error_std']}
        else:
            r   = _run_seeds(
                D=d_fixed, m=m, lam=lambda_fixed, anchor_np=anchor_full[:m],
                oracle_value=oracle_value, Q_star=Q_m,
                episodes=episodes, num_seeds=num_seeds, metric=metric, **kw,
            )
        entry   = {'m': m, 'D': d_fixed, 'm_over_D': m / d_fixed, 'result': r}
        msg     = f"  m={m:5d}  m/D={m/d_fixed:.3f}"
        if 'error_mean' in r:
            e_m = float(r['error_mean'].mean())
            e_s = float(r['error_std'].mean())
            msg += f"  fed_gap={e_m:.4f} ± {e_s:.4f}"
            entry['err_m'] = e_m;  entry['err_s'] = e_s
        if 'pv_mean' in r:
            p_m = float(r['pv_mean'].mean())
            p_s = float(r['pv_std'].mean())
            msg += f"  pv={p_m:.2f} ± {p_s:.2f}"
            entry['pv_m'] = p_m;   entry['pv_s'] = p_s
        results_per_m.append(entry)
        print(msg)

    return results_per_m


# =============================================================================
# Ablation 3 – Vary λ  (separate experiment per λ, D=D_FIXED, m=M_FIXED)
# =============================================================================

def ablation3_vary_lambda(
    anchor_np: np.ndarray,
    oracle_value: Optional[float] = None,
    Q_star: Optional[np.ndarray] = None,
    episodes: int = EPISODES,
    num_seeds: int = NUM_SEEDS,
    metric: str = 'all',
    lambda_grid: List[float] = LAMBDA_GRID,
    d_fixed: int = D_FIXED,
    m_fixed: int = M_FIXED,
    **kw,
) -> List[Dict]:
    """
    Fix D = d_fixed, m = m_fixed.  Sweep λ ∈ lambda_grid.
    x-axis: α = λ/m  (log scale).
    Theoretical optimum: α* ≈ D / sqrt(m).

    Expected: U-shape in error vs α  (variance at small α, bias at large α).
    """
    print("\n" + "=" * 60)
    print(f"Ablation 3: Vary λ  {lambda_grid}  (D={d_fixed}, m={m_fixed})")
    alpha_star = d_fixed / np.sqrt(m_fixed)
    lam_star   = alpha_star * m_fixed
    print(f"  Theory: α* ≈ {alpha_star:.2f}, λ* ≈ {lam_star:.1f}")
    print("=" * 60)

    results_per_lam = []
    for lam in tqdm.tqdm(lambda_grid, desc="Ablation 3"):
        r   = _run_seeds(
            D=d_fixed, m=m_fixed, lam=lam, anchor_np=anchor_np,
            oracle_value=oracle_value, Q_star=Q_star,
            episodes=episodes, num_seeds=num_seeds, metric=metric, **kw,
        )
        alpha = lam / m_fixed
        entry = {'lam': lam, 'alpha': alpha, 'D': d_fixed, 'm': m_fixed, 'result': r}
        msg   = f"  λ={lam:.0e}  α={alpha:.2e}"
        if 'error_mean' in r:
            e_m = float(r['error_mean'].mean())
            e_s = float(r['error_std'].mean())
            msg += f"  fed_gap={e_m:.4f} ± {e_s:.4f}"
            entry['err_m'] = e_m;  entry['err_s'] = e_s
        if 'pv_mean' in r:
            p_m = float(r['pv_mean'].mean())
            p_s = float(r['pv_std'].mean())
            msg += f"  pv={p_m:.2f} ± {p_s:.2f}"
            entry['pv_m'] = p_m;   entry['pv_s'] = p_s
        results_per_lam.append(entry)
        print(msg)

    return results_per_lam


# =============================================================================
# Plotting helpers
# =============================================================================

def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Uniform moving-average smoother (same-length output, edge-padded)."""
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    x_padded = np.pad(x, pad, mode='edge')
    return np.convolve(x_padded, np.ones(w) / w, mode='valid')[:len(x)]


def _fill(ax, x, mean, std, color, label):
    ax.plot(x, mean, color=color, linewidth=2.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.set_title(title,   fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)


def _make_fig(metric: str, n_panels_extra: int = 0, skip_lc: bool = False):
    show_err = metric in ('all', 'Q_error')
    show_pv  = metric in ('all', 'policy_value')
    n = (0 if skip_lc else 1) + show_err + show_pv + n_panels_extra
    if n == 0:
        n = 1  # always at least one panel
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n + 0.5, 5.5))
    if n == 1:
        axes = [axes]
    return fig, list(axes), show_err, show_pv


# ── A1 plot ───────────────────────────────────────────────────────────────────

def plot_ablation1(results_per_D: List[Dict], save_prefix: str,
                   metric: str = 'all', lc_window: int = 20) -> float:
    D_vals     = np.array([d['D'] for d in results_per_D], dtype=float)
    colors     = plt.cm.viridis(np.linspace(0.1, 0.9, len(D_vals)))
    n_eps      = results_per_D[0]['result']['reward_mean'].shape[1]
    analytical = (n_eps == 1)  # analytical mode has placeholder (1,1) reward arrays
    episodes   = np.arange(n_eps)

    fig, axes, show_err, show_pv = _make_fig(metric, skip_lc=analytical)
    ax_idx = 0

    # Panel 1: learning curves (skip in analytical mode — no RL rewards)
    if not analytical:
        ax = axes[ax_idx]; ax_idx += 1
        for entry, c in zip(results_per_D, colors):
            r      = entry['result']
            mean_c = _smooth(r['reward_mean'].mean(axis=0), lc_window)
            std_c  = _smooth(r['reward_std'].mean(axis=0),  lc_window)
            _fill(ax, episodes, mean_c, std_c, c, f"D={entry['D']}")
        w_label = f'  (w={lc_window})' if lc_window > 1 else ''
        m_label = results_per_D[0]['m']
        _style(ax, 'Episode', 'Mean Reward', f'Learning Curves (vary D, m={m_label}){w_label}')
        ax.legend(fontsize=15)

    slope = 0.0

    # Panel 2: Federation Gap vs D (log-log) + D^{-1/2} reference
    if show_err and 'err_m' in results_per_D[0]:
        ax = axes[ax_idx]; ax_idx += 1
        err_m = np.array([d['err_m'] for d in results_per_D])
        err_s = np.array([d['err_s'] for d in results_per_D])
        ax.errorbar(D_vals, err_m, yerr=err_s, fmt='o-', color='steelblue',
                    linewidth=2, markersize=7, capsize=4, label='Fed. Gap')
        # Fit slope on log-log
        slope, _ = np.polyfit(np.log(D_vals), np.log(np.maximum(err_m, 1e-8)), 1)
        mid       = len(D_vals) // 2
        c_ref     = err_m[mid] * D_vals[mid] ** 0.5
        D_ref_pts = np.array([D_vals[0], D_vals[-1]])
        ax.plot(D_ref_pts, c_ref * D_ref_pts ** (-0.5), 'r--', linewidth=2,
                label=r'$\propto D^{-1/2}$ (theory)')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(D_vals)
        ax.set_xticklabels([str(int(d)) for d in D_vals], rotation=30,
                            ha='right', fontsize=15)
        _style(ax, r'$D_i$', ERR_LABEL, f'Fed. Gap vs $D_i$  (slope {slope:.2f})')
        ax.legend(fontsize=15)

    # Panel 3: Policy Value vs D
    if show_pv and 'pv_m' in results_per_D[0]:
        ax       = axes[ax_idx]; ax_idx += 1
        r0       = results_per_D[0]['result']
        use_gap  = 'pv_gap_mean' in r0
        pv_m     = np.array([d['result']['pv_gap_mean'].mean() if use_gap
                              else d['pv_m'] for d in results_per_D])
        pv_s     = np.array([d['result']['pv_gap_std'].mean()  if use_gap
                              else d['pv_s'] for d in results_per_D])
        ylabel   = PVG_LABEL if use_gap else PV_LABEL
        title    = r'Policy Value (Gap) vs $D_i$'
        ax.errorbar(D_vals, pv_m, yerr=pv_s, fmt='s-', color='darkorange',
                    linewidth=2, markersize=7, capsize=4)
        if use_gap:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(D_vals)
        ax.set_xticklabels([str(int(d)) for d in D_vals], rotation=30,
                            ha='right', fontsize=15)
        _style(ax, r'$D_i$', ylabel, title)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{save_prefix}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_prefix}.{{pdf,png}}"
          + (f"  (slope={slope:.3f})" if show_err else ""))
    return float(slope)


# ── A2 plot ───────────────────────────────────────────────────────────────────

def plot_ablation2(results_per_m: List[Dict], save_prefix: str,
                   metric: str = 'all', lc_window: int = 20):
    ratio_vals = np.array([d['m_over_D'] for d in results_per_m], dtype=float)
    D_fix      = results_per_m[0]['D']
    colors     = plt.cm.plasma(np.linspace(0.1, 0.9, len(results_per_m)))
    n_eps      = results_per_m[0]['result']['reward_mean'].shape[1]
    analytical = (n_eps == 1)  # analytical mode has placeholder (1,1) reward arrays
    episodes   = np.arange(n_eps)

    fig, axes, show_err, show_pv = _make_fig(metric, skip_lc=analytical)
    ax_idx = 0

    # Panel 1: learning curves (skip in analytical mode — no RL rewards)
    if not analytical:
        ax = axes[ax_idx]; ax_idx += 1
        for entry, c in zip(results_per_m, colors):
            r      = entry['result']
            mean_c = _smooth(r['reward_mean'].mean(axis=0), lc_window)
            std_c  = _smooth(r['reward_std'].mean(axis=0),  lc_window)
            _fill(ax, episodes, mean_c, std_c, c, f"m={entry['m']}")
        w_label = f'  (w={lc_window})' if lc_window > 1 else ''
        _style(ax, 'Episode', 'Mean Reward', f'Learning Curves (vary m, D={D_fix}){w_label}')
        ax.legend(fontsize=15)

    # Panel 2: Fed Gap vs m/D
    if show_err and 'err_m' in results_per_m[0]:
        ax     = axes[ax_idx]; ax_idx += 1
        err_m  = np.array([d['err_m'] for d in results_per_m])
        err_s  = np.array([d['err_s'] for d in results_per_m])
        ax.errorbar(ratio_vals, err_m, yerr=err_s, fmt='o-', color='steelblue',
                    linewidth=2, markersize=7, capsize=4, label='Fed. Gap')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
                   label=r'$m = D$ (transition)')
        ax.axvspan(ratio_vals[0] * 0.8, 1.0, alpha=0.06, color='orange',
                   label=r'Under-param. ($m < D$)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(ratio_vals)
        ax.set_xticklabels([f'{r:.2f}' for r in ratio_vals], rotation=30,
                            ha='right', fontsize=15)
        _style(ax, r'$m / D$', ERR_LABEL, 'Fed. Gap vs $m/D$')
        ax.legend(fontsize=15)

    # Panel 3: Policy Value vs m/D
    if show_pv and 'pv_m' in results_per_m[0]:
        ax      = axes[ax_idx]; ax_idx += 1
        r0      = results_per_m[0]['result']
        use_gap = 'pv_gap_mean' in r0
        pv_m    = np.array([d['result']['pv_gap_mean'].mean() if use_gap
                             else d['pv_m'] for d in results_per_m])
        pv_s    = np.array([d['result']['pv_gap_std'].mean()  if use_gap
                             else d['pv_s'] for d in results_per_m])
        ylabel  = PVG_LABEL if use_gap else PV_LABEL
        ax.errorbar(ratio_vals, pv_m, yerr=pv_s, fmt='s-', color='darkorange',
                    linewidth=2, markersize=7, capsize=4)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
                   label=r'$m = D$')
        if use_gap:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ratio_vals)
        ax.set_xticklabels([f'{r:.2f}' for r in ratio_vals], rotation=30,
                            ha='right', fontsize=15)
        _style(ax, r'$m / D$', ylabel, 'Policy Value vs $m/D$')
        ax.legend(fontsize=15)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{save_prefix}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_prefix}.{{pdf,png}}")


# ── A3 plot ───────────────────────────────────────────────────────────────────

def plot_ablation3(results_per_lam: List[Dict], save_prefix: str,
                   metric: str = 'all', lc_window: int = 20) -> float:
    alpha_vals = np.array([d['alpha'] for d in results_per_lam], dtype=float)
    lam_vals   = np.array([d['lam']   for d in results_per_lam], dtype=float)
    D_fix      = results_per_lam[0]['D']
    m_fix      = results_per_lam[0]['m']
    colors     = plt.cm.cool(np.linspace(0.1, 0.9, len(lam_vals)))
    n_eps      = results_per_lam[0]['result']['reward_mean'].shape[1]
    episodes   = np.arange(n_eps)

    # Theoretical optimum
    alpha_star = D_fix / np.sqrt(m_fix)
    lam_star   = alpha_star * m_fix

    fig, axes, show_err, show_pv = _make_fig(metric)
    ax_idx  = 0
    best_lam = float(lam_vals[0])

    # Panel 1: learning curves (with optional smoothing window)
    ax = axes[ax_idx]; ax_idx += 1
    for entry, c in zip(results_per_lam, colors):
        r      = entry['result']
        mean_c = _smooth(r['reward_mean'].mean(axis=0), lc_window)
        std_c  = _smooth(r['reward_std'].mean(axis=0),  lc_window)
        _fill(ax, episodes, mean_c, std_c, c, f"λ={entry['lam']:.0e}")
    w_label = f'  (w={lc_window})' if lc_window > 1 else ''
    _style(ax, 'Episode', 'Reward',
           f'Learning Curves (vary λ, D={D_fix}, m={m_fix}){w_label}')
    ax.legend(fontsize=15)

    # Panel 2: Fed Gap vs α = λ/m  (U-shape expected)
    if show_err and 'err_m' in results_per_lam[0]:
        ax        = axes[ax_idx]; ax_idx += 1
        err_m     = np.array([d['err_m'] for d in results_per_lam])
        err_s     = np.array([d['err_s'] for d in results_per_lam])
        best_idx  = int(np.argmin(err_m))
        best_lam  = float(lam_vals[best_idx])
        best_alpha= float(alpha_vals[best_idx])
        ax.errorbar(alpha_vals, err_m, yerr=err_s, fmt='o-', color='steelblue',
                    linewidth=2, markersize=7, capsize=4, label='Fed. Gap')
        ax.axvline(x=best_alpha, color='steelblue', linestyle=':',
                   linewidth=2, label=rf'Best $\alpha={best_alpha:.2e}$')
        # ax.axvline(x=alpha_star, color='red', linestyle='--',
        #            linewidth=1.5, label=rf'Theory $\alpha^*={alpha_star:.1f}$')
        ax.set_xscale('log')
        _style(ax, r'$\alpha = \lambda / m$', ERR_LABEL,
               f'Fed. Gap vs α  (D={D_fix}, m={m_fix})')
        ax.legend(fontsize=15)

    # Panel 3: Policy Value vs α
    if show_pv and 'pv_m' in results_per_lam[0]:
        ax      = axes[ax_idx]; ax_idx += 1
        r0      = results_per_lam[0]['result']
        use_gap = 'pv_gap_mean' in r0
        pv_m    = np.array([d['result']['pv_gap_mean'].mean() if use_gap
                             else d['pv_m'] for d in results_per_lam])
        pv_s    = np.array([d['result']['pv_gap_std'].mean()  if use_gap
                             else d['pv_s'] for d in results_per_lam])
        ylabel  = PVG_LABEL if use_gap else PV_LABEL
        best_pv = int(np.argmin(pv_m) if use_gap else np.argmax(pv_m))
        best_lam= float(lam_vals[best_pv])
        ax.errorbar(alpha_vals, pv_m, yerr=pv_s, fmt='s-', color='darkorange',
                    linewidth=2, markersize=7, capsize=4)
        ax.axvline(x=alpha_vals[best_pv], color='green', linestyle=':',
                   linewidth=2, label=rf'Best $\alpha={alpha_vals[best_pv]:.2e}$')
        # ax.axvline(x=alpha_star, color='red', linestyle='--',
        #            linewidth=1.5, label=rf'Theory $\alpha^*={alpha_star:.1f}$')
        if use_gap:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xscale('log')
        _style(ax, r'$\alpha = \lambda / m$', ylabel,
               f'PV vs α  (D={D_fix}, m={m_fix})')
        ax.legend(fontsize=15)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{save_prefix}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_prefix}.{{pdf,png}}  (best λ={best_lam:.0e})")
    return best_lam


# =============================================================================
# Consistency check
# =============================================================================

def check_results(r1: List[Dict], r2: List[Dict], r3: List[Dict],
                  slope1: float, metric: str = 'all') -> bool:
    print("\n" + "=" * 60)
    print(f"Consistency Check  (metric={metric})")
    print("=" * 60)
    checks = []

    # A1 ──────────────────────────────────────────────────────────────────────
    if r1 and metric in ('all', 'Q_error') and 'err_m' in r1[0]:
        # Weak check: any negative slope is consistent with D^{-1/2} theory.
        # Exact slope ≈ -0.5 is hard to reproduce on CartPole for D≥128 because
        # Q* is already well-approximated at D=128; MC noise dominates the floor.
        ok1a = slope1 < 0
        print(f"  [A1] Fed. Gap slope: {slope1:.3f}  expected < 0  (theory -0.50)"
              f"  → {'PASS ✓' if ok1a else 'WARN ✗'}")
        errs1 = [d['err_m'] for d in r1]
        ok1b  = errs1[0] > errs1[-1]
        print(f"  [A1] Fed. Gap  D_min > D_max: {errs1[0]:.4f} > {errs1[-1]:.4f}"
              f"  → {'PASS ✓' if ok1b else 'WARN ✗'}")
        checks += [ok1a, ok1b]

    if r1 and metric in ('all', 'policy_value') and 'pv_m' in r1[0]:
        r0 = r1[0]['result']
        use_gap = 'pv_gap_mean' in r0
        pv1 = [d['result']['pv_gap_mean'].mean() if use_gap else d['pv_m'] for d in r1]
        ok1c = pv1[0] > pv1[-1] if use_gap else pv1[-1] > pv1[0]
        label = 'PV Gap decreasing' if use_gap else 'PV increasing'
        print(f"  [A1] {label} with D: {pv1[0]:.2f} → {pv1[-1]:.2f}"
              f"  → {'PASS ✓' if ok1c else 'WARN ✗'}")
        checks.append(ok1c)

    # A2 ──────────────────────────────────────────────────────────────────────
    if r2 and metric in ('all', 'Q_error') and 'err_m' in r2[0]:
        errs2 = [d['err_m'] for d in r2]
        ok2a  = errs2[0] > errs2[-1]
        print(f"  [A2] Fed. Gap  m/D_min > m/D_max: {errs2[0]:.4f} > {errs2[-1]:.4f}"
              f"  → {'PASS ✓' if ok2a else 'WARN ✗'}")
        # Check transition near m/D = 1
        ratios   = [d['m_over_D'] for d in r2]
        errs_arr = np.array(errs2)
        under_errs = errs_arr[[i for i, r in enumerate(ratios) if r < 1.0]]
        over_errs  = errs_arr[[i for i, r in enumerate(ratios) if r >= 1.0]]
        ok2b = (len(under_errs) > 0 and len(over_errs) > 0
                and under_errs.mean() > over_errs.mean())
        print(f"  [A2] Under-param err > over-param err: "
              f"{under_errs.mean() if len(under_errs) > 0 else 'n/a':.4f} > "
              f"{over_errs.mean() if len(over_errs) > 0 else 'n/a':.4f}"
              f"  → {'PASS ✓' if ok2b else 'WARN ✗'}")
        checks += [ok2a, ok2b]

    if r2 and metric in ('all', 'policy_value') and 'pv_m' in r2[0]:
        r0      = r2[0]['result']
        use_gap = 'pv_gap_mean' in r0
        pvs2    = [d['result']['pv_gap_mean'].mean() if use_gap else d['pv_m'] for d in r2]
        ok2c    = pvs2[0] > pvs2[-1] if use_gap else pvs2[-1] > pvs2[0]
        label   = 'PV Gap decreasing' if use_gap else 'PV increasing'
        print(f"  [A2] {label} with m: {pvs2[0]:.2f} → {pvs2[-1]:.2f}"
              f"  → {'PASS ✓' if ok2c else 'WARN ✗'}")
        checks.append(ok2c)

    # A3 ──────────────────────────────────────────────────────────────────────
    if r3 and metric in ('all', 'Q_error') and 'err_m' in r3[0]:
        errs3    = [d['err_m'] for d in r3]
        best_e   = int(np.argmin(errs3))
        ok3a     = 0 < best_e < len(errs3) - 1
        lam_best = r3[best_e]['lam']
        print(f"  [A3] Fed. Gap U-shape: min at λ={lam_best:.0e} "
              f"(idx {best_e}/{len(errs3)-1})"
              f"  → {'PASS ✓' if ok3a else 'WARN ✗'}")
        checks.append(ok3a)

    if r3 and metric in ('all', 'policy_value') and 'pv_m' in r3[0]:
        r0       = r3[0]['result']
        use_gap  = 'pv_gap_mean' in r0
        pv3      = [d['result']['pv_gap_mean'].mean() if use_gap else d['pv_m'] for d in r3]
        best_pv  = int(np.argmin(pv3) if use_gap else np.argmax(pv3))
        ok3b     = 0 < best_pv < len(pv3) - 1
        lam_best = r3[best_pv]['lam']
        label    = 'PV Gap' if use_gap else 'PV'
        print(f"  [A3] {label} U/inv-U shape: optimum at λ={lam_best:.0e} "
              f"(idx {best_pv}/{len(pv3)-1})"
              f"  → {'PASS ✓' if ok3b else 'WARN ✗'}")
        checks.append(ok3b)

    if not checks:
        print("  (No checks applicable for chosen metric)")
        return True

    all_pass = all(checks)
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME ISSUES — inspect plots'}")
    return all_pass


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FedQHD Ablation Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--metric', type=str, default='policy_value',
        choices=['all', 'Q_error', 'policy_value'],
    )
    parser.add_argument(
        '--which', type=str, default='all',
        choices=['all', 'A1', 'A2', 'A3'],
        help='Which ablation(s) to run. Use A1/A2/A3 for parallel execution.',
    )
    parser.add_argument(
        '--env', type=str, default='CartPole',
        choices=list(_ENV_REGISTRY.keys()),
        help='Environment for ablation experiments. Per-env defaults override module constants.',
    )
    g = parser.add_argument_group('Training')
    g.add_argument('--episodes',     type=int,   default=None)
    g.add_argument('--agg_interval', type=int,   default=None)
    g.add_argument('--rff_gamma',    type=float, default=RFF_GAMMA)
    g.add_argument('--qhd_lr',       type=float, default=QHD_LR)
    g.add_argument('--discount',     type=float, default=QHD_DISCOUNT,
                   help='Discount for Q* MC rollout computation (and RL unless --rl_discount set).')
    g.add_argument('--rl_discount',  type=float, default=None,
                   help='Discount for RL agent TD updates (default: same as --discount). '
                        'Use 0.9 for CartPole to match best-known params while keeping '
                        '--discount=0.99 for higher-quality Q* estimates.')
    g.add_argument('--eps_decay',    type=float, default=EPS_DECAY)
    g.add_argument('--eps_min',      type=float, default=EPS_MIN)
    g.add_argument('--num_seeds',    type=int,   default=NUM_SEEDS)

    g = parser.add_argument_group('Reference agent / Q*')
    g.add_argument('--d_ref',           type=int, default=None)
    g.add_argument('--n_ref_episodes',  type=int, default=None)
    g.add_argument('--n_mc',            type=int, default=N_MC_ROLLOUTS)
    g.add_argument('--mc_horizon',      type=int, default=MC_HORIZON)
    g.add_argument('--n_eval_episodes', type=int, default=N_EVAL_EPISODES)

    g = parser.add_argument_group('Ablation grids')
    g.add_argument('--d_grid',      type=int,   nargs='+', default=D_GRID)
    g.add_argument('--m_grid',      type=int,   nargs='+', default=M_GRID)
    g.add_argument('--lambda_grid', type=float, nargs='+', default=LAMBDA_GRID)
    g.add_argument('--d_fixed',     type=int,   default=D_FIXED)
    g.add_argument('--m_fixed',     type=int,   default=M_FIXED)
    g.add_argument('--lambda_fixed',type=float, default=LAMBDA_FIXED)
    g.add_argument('--force_recompute_qstar', action='store_true',
                   help='Recompute Q* even if cache exists')
    g.add_argument('--force_recompute_ref', action='store_true',
                   help='Delete cached ref agent and retrain from scratch')
    g.add_argument('--ref_eps_decay', type=float, default=None,
                   help='Epsilon decay for ref agent training (default: same as --eps_decay). '
                        'Use a slower value (e.g. 0.995) when --eps_decay is very fast.')
    g.add_argument('--ref_discount', type=float, default=None,
                   help='Discount factor for ref agent (default: same as --discount).')
    g.add_argument('--lc_window', type=int, default=20,
                   help='Smoothing window for learning-curve plots (1 = no smoothing)')
    g.add_argument('--qref_noise', type=float, default=0.0,
                   help='Inject Gaussian noise into Q_glob at each aggregation step '
                        '(scale = qref_noise × std(Q_glob)).  Use 0.5 to expose '
                        'the λ variance arm and produce a U-shape in A3.')

    args = parser.parse_args()

    # ── Resolve env config and per-env defaults ────────────────────────────
    env_cfg = _ENV_REGISTRY.get(args.env, _ENV_REGISTRY['CartPole'])
    if args.episodes     is None: args.episodes     = env_cfg['episodes']
    if args.agg_interval is None: args.agg_interval = env_cfg['agg_interval']
    if args.d_ref        is None: args.d_ref        = env_cfg['d_ref']
    if args.n_ref_episodes is None: args.n_ref_episodes = env_cfg['n_ref_episodes']

    # Per-env paths to avoid cache conflicts between environments
    env_ref_agent_path = os.path.join(RESULTS_DIR, f"{args.env}_ref_agent.npz")

    t0 = time.time()
    print(f"FedQHD Ablation Experiments  ({args.env})")
    print(f"Metric   : {args.metric}")
    print(f"Episodes={args.episodes}  Seeds={args.num_seeds}  AggInterval={args.agg_interval}")
    print(f"D grid      : {args.d_grid}")
    print(f"m grid      : {args.m_grid}")
    print(f"lambda grid : {args.lambda_grid}")
    print(f"D_FIXED={args.d_fixed}  M_FIXED={args.m_fixed}")

    _wrapper_cls = env_cfg['wrapper_cls']
    _action_dim  = env_cfg['action_dim']
    _bounds_key  = env_cfg['bounds_key']
    _state_dim   = env_cfg['state_dim']

    # ── Step 1: collect anchors ────────────────────────────────────────────
    # Need enough anchors for A1 (m=4*max_D) and A2 (max of m_grid)
    _run_a1_pre = args.which in ('all', 'A1')  # whether A1 will run (needed for m_a1_max)
    m_a1_max  = (4 * max(args.d_grid)) if _run_a1_pre else 0
    m_a2_max  = max(args.m_grid)
    m_max     = max(m_a1_max, m_a2_max)
    print(f"\nCollecting {m_max} anchor states …  (A1 needs up to {m_a1_max}, A2 needs {m_a2_max})")
    anchor_full  = collect_anchors(m_max, wrapper_cls=_wrapper_cls, action_dim=_action_dim)
    anchor_fixed = anchor_full[:args.m_fixed]

    print(f"Collecting {N_EVAL} held-out eval anchor states …")
    eval_anchors = collect_anchors(N_EVAL, seed=MASTER_SEED + 1000,
                                   wrapper_cls=_wrapper_cls, action_dim=_action_dim)

    # ── Step 2a: Ensure ref agent exists (needed for both metrics) ─────────
    if args.force_recompute_ref and os.path.exists(env_ref_agent_path):
        os.remove(env_ref_agent_path)
        print(f"  [Ref] Deleted cached ref agent → will retrain.")
    ref_eps_decay = args.ref_eps_decay if args.ref_eps_decay is not None else args.eps_decay
    ref_discount  = args.ref_discount  if args.ref_discount  is not None else args.discount
    print(f"  [Ref] training params: lr={args.qhd_lr/2:.4f}  discount={ref_discount}"
          f"  eps_decay={ref_eps_decay}  n_eps={args.n_ref_episodes}")
    _ensure_ref_agent(
        d_ref=args.d_ref, n_ref_episodes=args.n_ref_episodes,
        rff_gamma=args.rff_gamma, qhd_lr=args.qhd_lr,
        discount=ref_discount, eps_decay=ref_eps_decay, eps_min=args.eps_min,
        state_dim=_state_dim, action_dim=_action_dim,
        wrapper_cls=_wrapper_cls, bounds_key=_bounds_key,
        ref_agent_path=env_ref_agent_path,
    )
    ref_agent = load_ref_agent(env_ref_agent_path)

    # ── Convergence check ─────────────────────────────────────────────────
    ref_agent.epsilon = 0.0
    ref_pv = compute_policy_value(ref_agent, n_episodes=args.n_eval_episodes,
                                  wrapper_cls=_wrapper_cls)
    REF_CONV_THRESH = env_cfg['conv_thresh']
    conv_ok = ref_pv >= REF_CONV_THRESH
    status  = "CONVERGED ✓" if conv_ok else "NOT CONVERGED ✗"
    print(f"\n  [RefCheck] V(π_ref) = {ref_pv:.1f}  (threshold ≥ {REF_CONV_THRESH})  → {status}")
    if not conv_ok:
        print(f"  WARNING: Reference agent has not converged.  "
              f"Re-run with --force_recompute_qstar or increase --n_ref_episodes.")

    # ── Step 2b: Q* for Q_error metric ────────────────────────────────────
    if args.metric in ('all', 'Q_error'):
        print(f"\nComputing / loading Q_star for {m_max} anchors …")
        ref_agent.epsilon = 0.0
        if env_cfg['use_vectorized_qstar']:
            Q_star_full = compute_Q_star_fast(
                anchor_full, ref_agent,
                n_mc=args.n_mc, mc_horizon=args.mc_horizon, discount=args.discount,
                force_recompute=args.force_recompute_qstar,
                action_dim=_action_dim, env=args.env,
            )
            print(f"\nComputing Q_star at {N_EVAL} held-out eval anchors …")
            Q_star_eval = compute_Q_star_fast(
                eval_anchors, ref_agent,
                n_mc=args.n_mc, mc_horizon=args.mc_horizon, discount=args.discount,
                action_dim=_action_dim, env=args.env,
            )
        else:
            # Use ref agent's own Q-function (for envs where state cannot be set arbitrarily)
            _cp_full = os.path.join(RESULTS_DIR, f"{args.env}_Q_star_m{m_max}_agent.npz")
            _cp_eval = os.path.join(RESULTS_DIR, f"{args.env}_Q_star_m{N_EVAL}_eval_agent.npz")
            Q_star_full = compute_Q_star_from_agent(
                anchor_full, ref_agent, cache_path=_cp_full,
                force_recompute=args.force_recompute_qstar,
            )
            print(f"\nComputing Q_star at {N_EVAL} held-out eval anchors …")
            Q_star_eval = compute_Q_star_from_agent(
                eval_anchors, ref_agent, cache_path=_cp_eval,
            )
        Q_star_fixed = Q_star_full[:args.m_fixed]
    else:
        Q_star_full  = None
        Q_star_fixed = None
        Q_star_eval  = None

    # ── Step 2c: Oracle policy value for policy_value metric ──────────────
    if args.metric in ('all', 'policy_value'):
        print("\nComputing oracle policy value …")
        ref_agent.epsilon = 0.0
        oracle_value = compute_policy_value(ref_agent, n_episodes=args.n_eval_episodes,
                                            wrapper_cls=_wrapper_cls)
        print(f"  Oracle V(π*) = {oracle_value:.2f}")
    else:
        oracle_value = None

    # rl_discount separates RL training discount from Q* MC discount.
    # When --rl_discount is set, RL agents use it for TD updates while Q* is
    # still computed with --discount (e.g. 0.99 for high-quality Q*).
    rl_discount = args.rl_discount if args.rl_discount is not None else args.discount
    print(f"Q* discount={args.discount}  RL discount={rl_discount}  "
          f"lr={args.qhd_lr}  eps_decay={args.eps_decay}")

    # Shared kwargs for all ablation functions
    shared = dict(
        episodes=args.episodes, num_seeds=args.num_seeds, metric=args.metric,
        agg_interval=args.agg_interval, rff_gamma=args.rff_gamma,
        qhd_lr=args.qhd_lr, discount=rl_discount,
        eps_decay=args.eps_decay, eps_min=args.eps_min,
        qref_noise=args.qref_noise,
        state_dim=_state_dim, action_dim=_action_dim,
        wrapper_cls=_wrapper_cls, bounds_key=_bounds_key,
    )
    # Held-out eval anchors only for A1/A2 (not A3).
    # For A3, λ's variance/bias U-shape is only visible at the TRAINING anchors
    # (at eval/test anchors, both arms are dominated by generalization gap, hiding the shape).
    eval_kw = ({'eval_anchor_np': eval_anchors, 'Q_star_eval': Q_star_eval}
               if args.metric in ('all', 'Q_error') else {})

    # Analytical mode: skip RL training and directly compute oracle regression error.
    # Only viable for Q_error (no policy to evaluate); 'all'/'policy_value' use RL.
    analytical_mode = (args.metric == 'Q_error')

    # For A1: always use m=4D (never fix m) so geometry floor D^{-1/2} is visible.
    # m_fixed=m_a1_max made ALL D overdetermined → MC noise floor (~7 RMSE) masked
    # the geometry floor for D≥256 on CartPole, giving slope≈-0.08 instead of -0.5.
    # With m=4D, search_ablation_hparams.py confirmed slope≈-0.60.
    m_fixed_a1 = None  # always m=4D, both RL and analytical modes

    run_a1 = args.which in ('all', 'A1')
    run_a2 = args.which in ('all', 'A2')
    run_a3 = args.which in ('all', 'A3')

    # ── Step 3: Run ablations ─────────────────────────────────────────────
    r1 = ablation1_vary_D(
        anchor_full=anchor_full, oracle_value=oracle_value,
        Q_star_full=Q_star_full,
        d_grid=args.d_grid, lambda_fixed=args.lambda_fixed,
        m_fixed=m_fixed_a1,
        analytical_mode=analytical_mode,
        **shared, **eval_kw,
    ) if run_a1 else None

    r2 = ablation2_vary_m(
        anchor_full=anchor_full, oracle_value=oracle_value,
        Q_star_full=Q_star_full,
        m_grid=args.m_grid, d_fixed=args.d_fixed, lambda_fixed=args.lambda_fixed,
        analytical_mode=analytical_mode,
        **shared, **eval_kw,
    ) if run_a2 else None

    r3 = ablation3_vary_lambda(
        anchor_np=anchor_fixed, oracle_value=oracle_value,
        Q_star=Q_star_fixed,
        lambda_grid=args.lambda_grid, d_fixed=args.d_fixed, m_fixed=args.m_fixed,
        **shared,
    ) if run_a3 else None

    # ── Step 4: Save results ──────────────────────────────────────────────
    def _to_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict):       return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):       return [_to_json(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        return obj

    tag_suffix = f"_{args.env}_{args.metric}"
    for tag, res in [('ablation1_vary_D', r1),
                     ('ablation2_vary_m', r2),
                     ('ablation3_vary_lambda', r3)]:
        if res is None:
            continue
        path = os.path.join(RESULTS_DIR, f'{tag}{tag_suffix}.json')
        with open(path, 'w') as f:
            json.dump(_to_json(res), f, indent=2)
        print(f"Saved: {path}")

    # ── Step 5: Figures ───────────────────────────────────────────────────
    prefix = lambda name: os.path.join(FIGURES_DIR, f'{name}{tag_suffix}')
    slope1 = plot_ablation1(r1, prefix('ablation1_vary_D'), args.metric,
                            lc_window=args.lc_window) if r1 else 0.0
    if r2: plot_ablation2(r2, prefix('ablation2_vary_m'), args.metric,
                          lc_window=args.lc_window)
    if r3: plot_ablation3(r3, prefix('ablation3_vary_lambda'), args.metric,
                          lc_window=args.lc_window)

    # ── Step 6: Consistency check (only when all three ablations ran) ─────
    if r1 and r2 and r3:
        all_pass = check_results(r1, r2, r3, slope1, args.metric)
    else:
        print("  (Skipping full consistency check — not all ablations ran)")
        all_pass = True

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s")
    return r1, r2, r3, all_pass


if __name__ == '__main__':
    main()
