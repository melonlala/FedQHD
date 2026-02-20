import gymnasium as gym
import numpy as np
from env.gridworld import GridWorld 

class GridWorldWrapper:
    """
    support Gymnasium interface for GridWorld
    """
    def __init__(self, size):
        self.env = GridWorld(size=size)
        self.state_dim = 2 # (x, y)
        self.action_dim = 4
        
    def reset(self):
        state = self.env.reset()
        return state, {} 
    
    def step(self, action):
        next_state, reward, done = self.env.step(action)
        return next_state, reward, done, False, {}
    
    def is_success(self, state):
        return self.env.is_success(state)

class LunarLanderWrapper:
    """
    support Gymnasium interface for LunarLander
    """
    def __init__(self, render_mode=None, max_episode_steps=None):
        kwargs = {"render_mode": render_mode}
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        self.env = gym.make("LunarLander-v3", **kwargs)
        self.state_dim = self.env.observation_space.shape[0] # 8维
        self.action_dim = self.env.action_space.n          # 4个动作
        self.episode_reward = 0 # Inner variable to track total reward

    def reset(self):
        self.episode_reward = 0 # Reset total reward
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # Accumulate reward
        self.episode_reward += reward
        done = terminated or truncated
        return next_state, reward, done, truncated, info

    def is_success(self, state=None):
        """
        Define success as achieving a total episode reward of 200 or more
        in LunarLander environment.
        """
        return self.episode_reward >= 200

    def close(self):
        self.env.close()

class MountainCarWrapper:
    """
    support Gymnasium interface for MountainCar
    """
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.state_dim = self.env.observation_space.shape[0] # 2维
        self.action_dim = self.env.action_space.n          # 3个动作

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, truncated, info

    def is_success(self, state):
        # Define success as reaching the flag at position >= 0.5
        return state[0] >= 0.5

class CliffWalkingWrapper:
    """
    Support Gymnasium interface for CliffWalking.
    Fixes unpacking issues with reset() and step() to ensure integer states.
    """
    def __init__(self):
        self.env = gym.make("CliffWalking-v1")
        self.state_dim = 2 # (row, col) representation
        # self.state_num = self.env.observation_space.n 
        self.state_space = (4, 12)  # Grid size: 4 rows, 12 columns
        self.action_dim = self.env.action_space.n      # 4 actions
        self.max_steps = 5000
        self.current_step = 0

    def reset(self):
        # We must unpack this so 'state' is just the integer index, not a tuple.
        state, info = self.env.reset()
        self.current_step = 0
        
        # return state as (x, y) tuple
        return (state // 12, state % 12), info

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.current_step += 1
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            truncated = True
            info['TimeLimit.truncated'] = True
        # 'done' is usually true if terminated (goal/cliff) or truncated (time limit)
        done = terminated or truncated

        # return next_state as (x, y) tuple
        return (next_state // 12, next_state % 12), reward, done, truncated, info

    def is_success(self, state):
        # Fix 3: Standard CliffWalking-v0 does not have an is_success method.
        # The goal state is the bottom-right cell, which is index 47.
        return state == (3, 11)

    def close(self):
        self.env.close()

class CartPoleWrapper:
    """
    support Gymnasium interface for CartPole
    """
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape[0] # 4维
        self.action_dim = self.env.action_space.n          # 2个动作
        self.episode_reward = 0 # Inner variable to track total reward

    def reset(self):
        self.episode_reward = 0 # Reset total reward
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated
        return next_state, reward, done, truncated, info

    def is_success(self, state):
        # Define success as achieving a total episode reward of 475 or more
        return self.episode_reward >= 475
    
class AcrobotWrapper:
    """
    support Gymnasium interface for Acrobot
    """
    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        self.state_dim = self.env.observation_space.shape[0] # 6维
        self.action_dim = self.env.action_space.n          # 3个动作

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, truncated, info

    def is_success(self, state):
        # Define success as reaching the goal (top position)
        return state[0] >= 1.0

class TaxiWrapper:
    """
    support Gymnasium interface for Taxi
    Taxi-v3 has a discrete state space (500 states) representing:
    - Taxi position (5x5 grid = 25 positions)
    - Passenger location (5 locations: R, G, Y, B, or in taxi)
    - Destination (4 locations: R, G, Y, B)
    """
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        # For discrete agents (q_learning), use the raw discrete space
        self.state_space = self.env.observation_space.n  # 500
        # For continuous agents, we'll decode and use 4D representation
        self.state_dim = 4  # [taxi_row, taxi_col, passenger_loc, destination]
        self.action_dim = self.env.action_space.n  # 6 actions
        self.episode_reward = 0

    def _decode_state(self, state):
        """
        Decode the discrete state into components for continuous representation.
        State encoding: state = taxi_row * 100 + taxi_col * 20 + passenger_loc * 4 + destination
        """
        # Decode according to Taxi-v3 encoding
        taxi_row = state // 100
        taxi_col = (state % 100) // 20
        passenger_loc = (state % 20) // 4
        destination = state % 4

        # Normalize to [0, 1] for continuous agents
        return np.array([
            taxi_row / 4.0,      # 5 rows (0-4) -> [0, 1]
            taxi_col / 4.0,      # 5 cols (0-4) -> [0, 1]
            passenger_loc / 4.0, # 5 locations (0-4) -> [0, 1]
            destination / 3.0    # 4 destinations (0-3) -> [0, 1]
        ])

    def reset(self):
        self.episode_reward = 0
        state, info = self.env.reset()
        # Return decoded continuous state for continuous agents
        return self._decode_state(state), info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated
        # Return decoded continuous state
        return self._decode_state(next_state), reward, done, truncated, info

    def is_success(self, state=None):
        """
        In Taxi, success is defined by receiving the +20 reward for successful dropoff.
        We'll track if we achieved a positive episode reward (successful delivery gives +20,
        but each step costs -1, so positive means we succeeded efficiently)
        """
        return self.episode_reward >= 0

    def close(self):
        self.env.close()


class AtariPongWrapper:
    """
    Support Gymnasium interface for Atari Pong (RAM observations).
    State: 128D uint8 RAM normalized to [0, 1]
    Actions: 6 (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
    Success: positive episode reward (winning the game)
    """
    def __init__(self):
        try:
            import ale_py
            gym.register_envs(ale_py)
        except:
            pass

        self.env = gym.make("ALE/Pong-ram-v5")
        self.state_dim = 128
        self.action_dim = self.env.action_space.n  # 6 actions
        self.episode_reward = 0

    def reset(self):
        self.episode_reward = 0
        state, info = self.env.reset()
        return state.astype(np.float32) / 255.0, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated
        return next_state.astype(np.float32) / 255.0, reward, done, truncated, info

    def is_success(self, state=None):
        return self.episode_reward > 0

    def close(self):
        self.env.close()


class AtariFreewayWrapper:
    """
    Support Gymnasium interface for Atari Freeway (RAM observations).
    State: 128D uint8 RAM normalized to [0, 1]
    Actions: 3 (NOOP, UP, DOWN)
    Success: positive episode reward (crossing the road)
    """
    def __init__(self):
        try:
            import ale_py
            gym.register_envs(ale_py)
        except:
            pass

        self.env = gym.make("ALE/Freeway-ram-v5")
        self.state_dim = 128
        self.action_dim = self.env.action_space.n  # 3 actions
        self.episode_reward = 0

    def reset(self):
        self.episode_reward = 0
        state, info = self.env.reset()
        return state.astype(np.float32) / 255.0, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated
        return next_state.astype(np.float32) / 255.0, reward, done, truncated, info

    def is_success(self, state=None):
        return self.episode_reward > 0

    def close(self):
        self.env.close()


def create_env(args):
    if args.env == 'GridWorld':
        env = GridWorldWrapper(size=args.grid_size)
        return env
    elif args.env == 'LunarLander':
        render_mode = "human" if hasattr(args, 'render') and args.render else None
        max_ep_steps = getattr(args, 'max_episode_steps', None)
        env = LunarLanderWrapper(render_mode=render_mode, max_episode_steps=max_ep_steps)
        return env

    elif args.env == 'MountainCar':
        env = MountainCarWrapper()
        return env

    elif args.env == 'CartPole':
        env = CartPoleWrapper()
        return env

    elif args.env == 'CliffWalking':
        env = CliffWalkingWrapper()
        return env
    elif args.env == 'Acrobot':
        env = AcrobotWrapper()
        return env
    elif args.env == 'Taxi':
        env = TaxiWrapper()
        return env
    elif args.env == 'Pong':
        env = AtariPongWrapper()
        return env
    elif args.env == 'Freeway':
        env = AtariFreewayWrapper()
        return env
    else:
        raise ValueError(f"Unknown environment: {args.env}")