import numpy as np
import random

class HDSARSAAgent:
    """
    Linear SARSA Agent using Random Fourier Features (RFF) for State Encoding.
    
    Attributes:
        Combination of:
        1. Feature Extractor: HDC/RFF (Random Fourier Features)
        2. Control Algorithm: SARSA (On-Policy TD Control)
    """
    
    def __init__(self, 
                 grid_size=(8, 8),
                 n_actions: int = 4,
                 hd_dim: int = 10000,
                 rff_gamma: float = 1.0,
                 learning_rate: float = 0.05,  # Linear approx usually needs smaller LR
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 exploration_min: float = 0.01,
                 random_seed: int = 42):

        self.state_dim = len(grid_size)
        self.n_actions = n_actions
        self.hd_dim = hd_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.rff_gamma = rff_gamma
        
        # --- 1. Initialize RFF Encoding Parameters ---
        self.random_state = random_seed
        self._initialize_rff_parameters()
        
        # --- 2. Initialize Weights ---
        # Instead of a Q-table, we have a weight vector for each action
        # Shape: (n_actions, hd_dim)
        # Q(s, a) = weights[a] dot phi(s)
        self.model_vectors = np.zeros((n_actions, hd_dim), dtype=np.float64)
        
        # --- 3. SARSA Specific ---
        # To store the action selected for the next state, needed for the update
        self.next_action = None

    def _initialize_rff_parameters(self):
        """Initialize RFF parameters (Omega and Bias)"""
        np.random.seed(self.random_state)
        # sigma = sqrt(2 * gamma) is a common heuristic for RBF kernel approximation
        sigma_omega = np.sqrt(2 * self.rff_gamma)
        
        # Projection matrix Omega: (D, d)
        self.omega = np.random.normal(
            loc=0.0, 
            scale=sigma_omega, 
            size=(self.hd_dim, self.state_dim)
        )
        # Bias b: (D,) uniformly from 0 to 2pi
        self.b = np.random.uniform(0, 2 * np.pi, size=self.hd_dim)
        
        # Pre-compute constant for cosine encoding normalization
        self.scale_factor = np.sqrt(2.0 / self.hd_dim)

    def encode_state(self, state: np.ndarray):
        """
        Encode state using Random Fourier Features.
        Formula: phi(x) = sqrt(2/D) * cos(Omega * x + b)
        """
        # Ensure state is an array
        state = np.array(state)
        
        # Project and Apply Non-linearity
        # (D, d) dot (d,) -> (D,)
        projection = np.dot(self.omega, state) + self.b
        encoding = self.scale_factor * np.cos(projection)
        
        return encoding
    
    def get_q_value(self, state: np.ndarray, action: int):
        """Calculate Q(s, a) = w_a^T * phi(s)"""
        state_hv = self.encode_state(state)
        return np.dot(state_hv, self.model_vectors[action])
    
    def get_all_q_values(self, state: np.ndarray):
        """Calculate Q(s, :) for all actions"""
        state_hv = self.encode_state(state)
        # (n_actions, hd_dim) dot (hd_dim,) -> (n_actions,)
        return np.dot(self.model_vectors, state_hv)
    
    def choose_action(self, state: np.ndarray):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = self.get_all_q_values(state)
            # Random tie-breaking for stability
            max_q = np.max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def update_model(self, state: np.ndarray, action: int, reward: float, 
                    next_state: np.ndarray, done: bool):
        """
        Linear SARSA Update Rule:
        w_a <- w_a + alpha * (Target - Q(s,a)) * phi(s)
        
        Target = R + gamma * Q(s', a')  [On-Policy]
        """
        
        # 1. Encode current state features: phi(s)
        state_hv = self.encode_state(state)
        
        # 2. Get current prediction: Q(s, a)
        current_q = np.dot(state_hv, self.model_vectors[action])
        
        # 3. Calculate Target
        if done:
            target = reward
        else:
            # SARSA Logic: We need the Q-value of the actual next action
            # If next_action hasn't been chosen yet (by the training loop), choose it now
            if self.next_action is None:
                self.next_action = self.choose_action(next_state)
            
            # Q(s', a')
            next_q = self.get_q_value(next_state, self.next_action)
            target = reward + self.gamma * next_q
        
        # 4. Calculate TD Error
        td_error = target - current_q
        
        # 5. Gradient Descent Update
        # Gradient of linear function w^T * phi is just phi
        self.model_vectors[action] += self.learning_rate * td_error * state_hv
        
        # Reset next_action if episode is done
        if done:
            self.next_action = None
        
    def decay_exploration(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)