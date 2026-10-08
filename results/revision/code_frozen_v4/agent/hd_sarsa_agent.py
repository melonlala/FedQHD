import numpy as np
import random

class HDSARSAAgent:
    def __init__(self, 
                 state_dim=8,
                 action_dim=4, 
                 hd_dim=10000,
                 rff_gamma=1.0,
                 learning_rate=0.05,    # Increased: Linear models need higher LR than Neural Nets
                 discount_factor=0.99,
                 exploration_rate=1.0,
                 exploration_decay=0.9995, # Slower decay
                 exploration_min=0.01,
                 random_seed=42,
                 state_bounds=None):

        self.state_dim = state_dim
        self.n_actions = action_dim
        self.hd_dim = hd_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.rff_gamma = rff_gamma
        
        self.random_state = random_seed
        self._initialize_rff_parameters()
        
        # Weights
        self.model_vectors = np.zeros((self.n_actions, self.hd_dim), dtype=np.float64)
            
        
        # SARSA: Store the action intended for the next step
        self.next_action = None

        # Bounds for LunarLander
        if state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
        else:
             self.state_bounds = np.array([[-1, 1]] * self.state_dim)

    def _initialize_rff_parameters(self):
        np.random.seed(self.random_state)
        sigma_omega = np.sqrt(2 * self.rff_gamma)
        self.omega = np.random.normal(loc=0.0, scale=sigma_omega, size=(self.hd_dim, self.state_dim))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.hd_dim)
        self.scale_factor = np.sqrt(2.0 / self.hd_dim)
        self.rff_parameters = {'omega': self.omega, 'b': self.b, 'scale_factor': self.scale_factor}

    def _normalize_state(self, state):
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
        norm_state = self._normalize_state(state)
        projection = np.dot(self.omega, norm_state) + self.b
        return self.scale_factor * np.cos(projection)
    
    def get_all_q_values(self, state: np.ndarray):
        state_hv = self.encode_state(state)
        return np.dot(self.model_vectors, state_hv)
    
    # def choose_action(self, state: np.ndarray):
    #     """
    #     SARSA FIX: If self.next_action was calculated in the update 
    #     step (s, a, r, s', a'), we MUST take a' now.
    #     """
    #     if self.next_action is not None:
    #         action = self.next_action
    #         self.next_action = None # Consumed
    #         return action

    #     # Standard Epsilon-Greedy
    #     if np.random.random() < self.epsilon:
    #         return np.random.randint(self.n_actions)
    #     else:
    #         q_values = self.get_all_q_values(state)
    #         max_q = np.max(q_values)
    #         best_actions = [i for i, q in enumerate(q_values) if q == max_q]
    #         return np.random.choice(best_actions)
    
    def choose_action(self, state: np.ndarray):
        # q_values = self.get_all_q_values(state)
        # q_shifted = q_values - np.max(q_values)  # For numerical stability
        # exp_q = np.exp(q_shifted / max(self.epsilon, 1e-5))  # Avoid division by zero
        # action_probs = exp_q / np.sum(exp_q)
        # action = np.random.choice(self.n_actions, p=action_probs)
        # return action

        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.get_all_q_values(state)
            max_q = np.max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update_model(self, state: np.ndarray, action: int, reward: float, 
                    next_state: np.ndarray, done: bool):
        
        state_hv = self.encode_state(state)
        current_q = np.dot(state_hv, self.model_vectors[action])
        
        if done:
            target = reward
            self.next_action = None # Reset for next episode
        else:
            self.next_action = self.choose_action(next_state)
            next_state_hv = self.encode_state(next_state)
            next_q = np.dot(next_state_hv, self.model_vectors[self.next_action])
            target = reward + self.gamma * next_q
        
        td_error = target - current_q
        
        # Update weights
        self.model_vectors[action] += self.learning_rate * td_error * state_hv

    def decay_exploration(self):
        """Decays epsilon based on decay rate and minimum value."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)