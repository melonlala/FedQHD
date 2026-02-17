import numpy as np
import random
from env.bounds import state_bounds

class LinearSARSAAgentStructured:
    """
    Linear SARSA Agent with Structured RBF Features

    Key improvements over random RBF:
    - Uses 64 structured RBF centers (8x8 grid) instead of 1000 random centers
    - Centers are evenly distributed across the normalized state space
    - Sigma is calculated based on grid spacing for proper coverage

    Based on successful MountainCar implementation:
    https://github.com/Ameyapores/MountainCar-SARSA
    """
    def __init__(self, state_dim=2, action_dim=3,
                 learning_rate=0.01, discount_factor=0.999,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.001,
                 num_rbf_per_dim=8, state_bounds=None, random_seed=42):
        """
        Args:
            state_dim: Dimensionality of state space (MountainCar: 2)
            action_dim: Number of actions (MountainCar: 3)
            num_rbf_per_dim: Number of RBF centers per dimension (8 → 8x8=64 total for 2D)
            state_bounds: Bounds for state normalization
        """
        self.state_dim = state_dim
        self.num_actions = action_dim

        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_min = exploration_min

        # --- Structured RBF Configuration ---
        self.num_rbf_per_dim = num_rbf_per_dim
        self.num_centers = num_rbf_per_dim ** state_dim  # 8^2 = 64 for MountainCar
        self.feature_dim = self.num_centers + 1  # +1 for bias

        # Calculate sigma based on grid spacing
        # With 8 centers in [-1, 1], spacing = 2/(8-1) ≈ 0.286
        # Sigma = spacing ensures neighboring RBFs overlap properly
        self.rbf_sigma = 2.0 / (num_rbf_per_dim - 1) if num_rbf_per_dim > 1 else 0.5

        # Initialize structured RBF centers
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.rbf_centers = self._create_structured_rbf_centers()

        # Initialize weights (Actions x Features)
        self.weights = np.zeros((self.num_actions, self.feature_dim))

        # SARSA-specific: store next action for on-policy updates
        self.next_action = None

        # Set state bounds for normalization
        if state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
        else:
            self.state_bounds = np.array([[-1, 1]] * self.state_dim)

    def _normalize_state(self, state):
        """
        Normalize state to [-1, 1] based on predefined bounds.
        Clips values outside bounds.
        """
        norm_state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            min_val, max_val = self.state_bounds[i]
            denom = (max_val - min_val)
            if denom == 0:
                norm_state[i] = 0.0
                continue
            # Scale to [-1, 1]
            norm_state[i] = 2 * (state[i] - min_val) / denom - 1

        # Handle non-finite values
        norm_state = np.nan_to_num(norm_state, nan=0.0, posinf=0.0, neginf=0.0)
        norm_state = np.clip(norm_state, -1.0, 1.0)
        return norm_state

    def _create_structured_rbf_centers(self):
        """
        Create RBF centers on a regular grid covering the state space.

        For MountainCar (2D state):
        - Creates 8x8 = 64 centers
        - Evenly distributed across [-1, 1] x [-1, 1] normalized space
        - Provides structured coverage of position x velocity space

        Returns:
            centers: (num_centers, state_dim) array of RBF center positions
        """
        # Create 1D grid for each dimension
        grids = []
        for dim in range(self.state_dim):
            # Evenly space centers across [-1, 1]
            grid_1d = np.linspace(-1, 1, self.num_rbf_per_dim)
            grids.append(grid_1d)

        # Create meshgrid for all dimensions
        mesh = np.meshgrid(*grids, indexing='ij')

        # Flatten and stack to get (num_centers, state_dim)
        centers = np.stack([m.flatten() for m in mesh], axis=1)

        return centers

    def _get_features(self, state):
        """
        Compute RBF feature vector: φ(s)
        φ_i(s) = exp(-||s - c_i||^2 / (2 * sigma^2))

        Each RBF center creates a Gaussian activation based on distance.
        """
        # Normalize state to [-1, 1]
        norm_state = self._normalize_state(state)

        features = np.zeros(self.feature_dim)

        # Vectorized distance computation
        diff = norm_state - self.rbf_centers  # (num_centers, state_dim)
        dist_sq = np.sum(diff**2, axis=1)     # (num_centers,)

        # Gaussian activation
        features[:self.num_centers] = np.exp(-dist_sq / (2 * self.rbf_sigma**2))

        # Bias term
        features[-1] = 1.0

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def get_q_value(self, state, action):
        """Compute Q(s, a) = w_a · φ(s)"""
        features = self._get_features(state)
        return np.dot(self.weights[action], features)

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action
            q_values = np.array([self.get_q_value(state, a) for a in range(self.num_actions)], dtype=float)

            # Handle non-finite values
            if not np.any(np.isfinite(q_values)):
                return random.randint(0, self.num_actions - 1)

            q_values = np.nan_to_num(q_values, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            max_q = np.max(q_values)

            # Break ties randomly
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            if not best_actions:
                return random.randint(0, self.num_actions - 1)

            return random.choice(best_actions)

    def update_model(self, state, action, reward, next_state, done):
        """
        SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        On-policy TD learning using the actual next action.
        """
        features = self._get_features(state)
        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            # Use stored next_action for on-policy SARSA
            if self.next_action is None:
                self.next_action = self.choose_action(next_state)
            next_q = self.get_q_value(next_state, self.next_action)
            target = reward + self.gamma * next_q

        # TD error
        td_error = target - current_q

        # Skip update if values are non-finite
        if not np.isfinite(td_error) or not np.all(np.isfinite(features)):
            if done:
                self.next_action = None
            return

        # Gradient descent update
        self.weights[action] += self.lr * td_error * features

        if done:
            self.next_action = None

    def decay_exploration(self):
        """Decay exploration rate after each episode"""
        self.exploration_rate = max(self.exploration_min,
                                   self.exploration_rate * self.exploration_rate_decay)

    def get_model_vector(self):
        """Return flattened weights for federated averaging"""
        return self.weights.flatten()

    def set_model_vector(self, vector):
        """Set weights from flattened vector (for federated averaging)"""
        self.weights = vector.reshape(self.num_actions, self.feature_dim)
