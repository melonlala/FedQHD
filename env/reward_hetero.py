from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from typing import List, Tuple



@dataclass
class RewardPerturbation:
    """Defines reward perturbation for an agent"""
    scale: float = 1.0          # Multiplicative scaling
    offset: float = 0.0         # Additive offset
    state_dependent: bool = False
    noise_std: float = 0.0      # Gaussian noise std


class HeterogeneousRewardWrapper(gym.Wrapper):
    """Wrapper to create heterogeneous rewards across agents"""
    
    def __init__(self, env, perturbation: RewardPerturbation, agent_id: int, seed: int = None):
        super().__init__(env)
        self.perturbation = perturbation
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)
        self.reward_history = []
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply heterogeneous reward transformation
        modified_reward = self._modify_reward(reward, obs, action)
        self.reward_history.append((reward, modified_reward))
        
        return obs, modified_reward, terminated, truncated, info
    
    def _modify_reward(self, reward, state, action):
        p = self.perturbation
        
        # Base transformation: r_i = scale * r + offset
        modified = p.scale * reward + p.offset
        
        # State-dependent perturbation (optional)
        if p.state_dependent and state is not None:
            # Add perturbation based on state features
            state_factor = np.sin(np.sum(state) * (self.agent_id + 1))
            modified += 0.1 * state_factor
        
        # Add noise
        if p.noise_std > 0:
            modified += self.rng.normal(0, p.noise_std)
            
        return modified


def compute_reward_heterogeneity(perturbations: List[RewardPerturbation], 
                                  env, num_samples: int = 1000) -> Tuple[float, float]:
    """
    Compute reward heterogeneity metric:
    ρ_r := max_{i,j∈[N]} ||r^(i) - r^(j)||_∞ / ||r||_∞
    
    where ||r||_∞ = sup_{s,a} |r(s,a)|
    
    Returns:
        rho_r: The heterogeneity metric (≤ 2 by triangle inequality)
        r_inf_norm: The infinity norm of the base reward
    """
    N = len(perturbations)
    
    # Sample rewards from environment
    base_rewards = []
    env_copy = gym.make(env.spec.id)
    
    for _ in range(num_samples):
        state, _ = env_copy.reset()
        action = env_copy.action_space.sample()
        _, reward, _, _, _ = env_copy.step(action)
        base_rewards.append(reward)
    
    base_rewards = np.array(base_rewards)
    r_inf_norm = np.max(np.abs(base_rewards))
    
    if r_inf_norm == 0:
        r_inf_norm = 1.0  # Avoid division by zero
    
    # Compute modified rewards for each agent
    agent_rewards = []
    for p in perturbations:
        modified = p.scale * base_rewards + p.offset
        if p.noise_std > 0:
            modified += np.random.normal(0, p.noise_std, size=len(base_rewards))
        agent_rewards.append(modified)
    
    # Compute max pairwise infinity norm difference
    max_diff = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = np.max(np.abs(agent_rewards[i] - agent_rewards[j]))
            max_diff = max(max_diff, diff)
    
    rho_r = max_diff / r_inf_norm
    
    return rho_r, r_inf_norm


def create_heterogeneous_perturbations(num_agents: int,
                                        heterogeneity_level: float) -> List[RewardPerturbation]:
    """
    Create reward perturbations that achieve target heterogeneity level.

    heterogeneity_level: Target ρ_r value (0 = homogeneous, up to 2)

    The heterogeneity is primarily controlled by scale differences:
    - Max scale difference = heterogeneity_level
    - Small offset to add minor variation
    - This keeps actual ρ_r close to target
    """
    perturbations = []

    for i in range(num_agents):
        # Distribute perturbations to achieve target heterogeneity
        # Agent 0 gets no perturbation, others get increasing perturbations
        if num_agents == 1:
            scale = 1.0
            offset = 0.0
        else:
            # Linear interpolation to achieve target heterogeneity
            t = i / (num_agents - 1) if num_agents > 1 else 0
            # Primary heterogeneity from scale: [1-h/2, 1+h/2]
            scale = 1.0 + heterogeneity_level * (t - 0.5)
            # Small offset for minor variation (reduced from 10 to 0.5)
            offset = heterogeneity_level * 0.5 * (t - 0.5)

        perturbations.append(RewardPerturbation(
            scale=scale,
            offset=offset,
            noise_std=heterogeneity_level * 0.05  # Reduced noise
        ))

    return perturbations


def create_heterogeneous_env(args, agent_id: int, perturbation: RewardPerturbation):
    """Create environment with heterogeneous reward for specific agent"""
    if args.env == 'LunarLander':
        env = gym.make('LunarLander-v3')
    elif args.env == 'CartPole':
        env = gym.make('CartPole-v1')
    elif args.env == 'MountainCar':
        env = gym.make('MountainCar-v0')
    elif args.env == 'Acrobot':
        env = gym.make('Acrobot-v1')
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    return HeterogeneousRewardWrapper(env, perturbation, agent_id)
