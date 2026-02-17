import numpy as np
from env.bounds import state_bounds
class QHDAgent:
    """QHD Agent using Random Fourier Features with cosine kernel"""
    
    def __init__(self, 
                 state_dim: int = 8,
                 action_dim: int = 4,
                 hd_dim: int = 10000,
                 rff_gamma: float = 1.0,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 exploration_min: float = 0.01,
                 random_seed: int = 42,
                 state_bounds=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hd_dim = hd_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.rff_gamma = rff_gamma
        
        # Initialize RFF parameters for state encoding
        self.random_state = random_seed
        self._initialize_rff_parameters()
        
        # Initialize action encoding vectors (one-hot like but in HD space)
        if state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
        else:
            self.state_bounds = np.array([[-1, 1]] * self.state_dim)
        self.action_vectors = self._initialize_action_vectors()
        
        # Initialize Q-function model vectors (one for each action), randomly normalized
        self.model_vectors = np.zeros((self.action_dim, self.hd_dim), dtype=np.float64)
        for i in range(self.action_dim):
            self.model_vectors[i] = np.random.randn(self.hd_dim)
            self.model_vectors[i] /= np.linalg.norm(self.model_vectors[i])

        # # Target network (for stability)
        # self.target_model_vectors = self.model_vectors.copy()
        # self.target_update_freq = 100
        # self.step_count = 0

    def _initialize_rff_parameters(self):
        """Initialize RFF parameters using your specific encoder setup"""
        np.random.seed(self.random_state)
        sigma_omega = np.sqrt(2 * self.rff_gamma)
        self.omega = np.random.normal(
            loc=0.0, 
            scale=sigma_omega, 
            size=(self.hd_dim, self.state_dim)
        )
        self.b = np.random.uniform(0, 2 * np.pi, size=self.hd_dim)
        
        print("RFF kernel parameters:")
        print("OMEGA:", self.omega.shape)
        print("B:", self.b.shape)

    def _initialize_action_vectors(self):
        """Initialize hyperdimensional action vectors"""
        # Since we're using real-valued RFF, create real action vectors
        action_vectors = np.zeros((self.action_dim, self.hd_dim), dtype=np.float64)
        np.random.seed(self.random_state + 1)  # Different seed for action vectors

        for i in range(self.action_dim):
            # Create random real-valued vectors for each action
            action_vectors[i] = np.random.randn(self.hd_dim)
            # Normalize to unit length
            action_vectors[i] = action_vectors[i] / np.linalg.norm(action_vectors[i])
        return action_vectors
    def normalize_state(self, state):
        """Normalize state to [-1, 1] based on predefined bounds"""
        state = np.array(state, dtype=np.float32)
        if len(state) != len(self.state_bounds): return state

        # Normalize all dimensions to [-1, 1]
        for i in range(len(state)):
            min_val, max_val = self.state_bounds[i]
            # Handle bounds that might be equal (though unlikely here)
            if max_val - min_val == 0:
                state[i] = 0
            else:
                state[i] = 2 * (state[i] - min_val) / (max_val - min_val) - 1
        return state
    
    def encode_state(self, state: np.ndarray):
        """
        Encode state using your specific RFF kernel encoder
        """
        # Use your RFF kernel encoding formula
        state = self.normalize_state(state)
        encoding = np.sqrt(2.0 / self.hd_dim) * np.cos(np.dot(self.omega, state) + self.b)
        
        return encoding
    
    def get_q_values(self, state: np.ndarray, use_target: bool = False):
        """Get Q-values for all actions given a state"""
        state_hv = self.encode_state(state)
        model_vectors = self.model_vectors
        
        q_values = np.zeros(self.action_dim)
        for action in range(self.action_dim):
            # Q-value is dot product for real vectors
            q_values[action] = np.dot(state_hv, model_vectors[action])
        
        return q_values
    
    # def choose_action(self, state: np.ndarray):
    #     """Select action using epsilon-greedy policy"""
    #     if np.random.random() < self.epsilon:
    #         return np.random.choice(self.action_dim)
    #     else:
    #         q_values = self.get_q_values(state)
    #         return np.argmax(q_values)
    def choose_action(self, state: np.ndarray):
        # q_values = self.get_q_values(state)
        # # Softmax action selection
        # q_shifted = q_values - np.max(q_values)  # For numerical stability
        # exp_q = np.exp(q_shifted / max(self.epsilon, 1e-5))  # Avoid division by zero
        # action_probs = exp_q / np.sum(exp_q)
        # action = np.random.choice(self.action_dim, p=action_probs)
        # return action

        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def update_model(self, state: np.ndarray, action: int, reward: float, 
                    next_state: np.ndarray, done: bool):
        """Update the QHD model using the Bellman equation"""
        
        # Encode current state
        state_hv = self.encode_state(state)
        
        # Predicted Q-value
        q_pred = np.dot(state_hv, self.model_vectors[action])
        
        # Target Q-value using Bellman equation
        if done:    
            q_target = reward
        else:
            next_q_values = self.get_q_values(next_state, use_target=True)
            q_target = reward + self.gamma * np.max(next_q_values)
        
        # Update rule from paper: M_A = M_A + β(q_true - q_pred) * S
        error = q_target - q_pred
        self.model_vectors[action] += self.learning_rate * error * state_hv
        
        # # Update target network periodically
        # self.step_count += 1
        # if self.step_count % self.target_update_freq == 0:
        #     self.target_model_vectors = self.model_vectors.copy()
        
    def decay_exploration(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

