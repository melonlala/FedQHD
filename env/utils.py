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
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
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
    Support Gymnasium interface for Atari Pong-v0/v4
    Pong has pixel observations (210x160x3), so we need preprocessing
    """
    def __init__(self):
        try:
            import ale_py
            gym.register_envs(ale_py)
        except:
            pass

        self.env = gym.make("ALE/Pong-v5")
        # Use RAM state instead of pixels for simpler learning
        # RAM has 128 bytes of state information
        self.state_dim = 128  # RAM size
        self.action_dim = self.env.action_space.n  # 6 actions
        self.episode_reward = 0
        self.use_ram = True  # Flag to use RAM instead of pixels

    def _preprocess_observation(self, obs):
        """
        Preprocess Atari observation
        For RAM mode: normalize to [0, 1]
        """
        if self.use_ram:
            # Normalize RAM values to [0, 1]
            return obs / 255.0
        return obs

    def reset(self):
        self.episode_reward = 0
        obs, info = self.env.reset()

        # Get RAM state if available
        if hasattr(self.env.unwrapped, 'ale'):
            state = self.env.unwrapped.ale.getRAM()
            state = self._preprocess_observation(state)
        else:
            state = obs.flatten()[:128]  # Fallback

        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated

        # Get RAM state
        if hasattr(self.env.unwrapped, 'ale'):
            next_state = self.env.unwrapped.ale.getRAM()
            next_state = self._preprocess_observation(next_state)
        else:
            next_state = obs.flatten()[:128]  # Fallback

        return next_state, reward, done, truncated, info

    def is_success(self, state=None):
        """
        In Pong, success is winning the game (positive score)
        """
        return self.episode_reward > 0

    def close(self):
        self.env.close()


class AtariFreewayWrapper:
    """
    Support Gymnasium interface for Atari Freeway
    Freeway: Cross the road avoiding cars
    """
    def __init__(self):
        try:
            import ale_py
            gym.register_envs(ale_py)
        except:
            pass

        self.env = gym.make("ALE/Freeway-v5")
        self.state_dim = 128  # RAM size
        self.action_dim = self.env.action_space.n  # 3 actions (up, down, noop)
        self.episode_reward = 0

    def _preprocess_observation(self, obs):
        """Normalize RAM values to [0, 1]"""
        return obs / 255.0

    def reset(self):
        self.episode_reward = 0
        obs, info = self.env.reset()

        # Get RAM state
        if hasattr(self.env.unwrapped, 'ale'):
            state = self.env.unwrapped.ale.getRAM()
            state = self._preprocess_observation(state)
        else:
            state = obs.flatten()[:128]

        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = terminated or truncated

        # Get RAM state
        if hasattr(self.env.unwrapped, 'ale'):
            next_state = self.env.unwrapped.ale.getRAM()
            next_state = self._preprocess_observation(next_state)
        else:
            next_state = obs.flatten()[:128]

        return next_state, reward, done, truncated, info

    def is_success(self, state=None):
        """
        In Freeway, success is crossing the road (positive reward)
        """
        return self.episode_reward > 0

    def close(self):
        self.env.close()


def create_env(args):
    if args.env == 'GridWorld':
        env = GridWorldWrapper(size=args.grid_size)
        return env
    elif args.env == 'LunarLander':
        render_mode = "human" if hasattr(args, 'render') and args.render else None
        env = LunarLanderWrapper(render_mode=render_mode)
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