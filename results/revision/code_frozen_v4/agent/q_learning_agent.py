import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_dim, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
        self.q_table = np.zeros((state_space[0] * state_space[1], action_dim))
        self.state_space = state_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_min = exploration_min
        self.num_actions = action_dim
    
    def choose_action(self, state):
        # cast state into int index
        state = state[0] * self.state_space[1] + state[1]
        
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update_model(self, state, action, reward, next_state, done):
        # cast states into int index
        state = int(state[0] * self.state_space[1] + state[1])
        next_state = int(next_state[0] * self.state_space[1] + next_state[1])
        max_future_q = np.max(self.q_table[next_state, :])
        
        if done:
            max_future_q = 0.0
            
        current_q = self.q_table[state, action]
        self.q_table[state, action] = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)

    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_rate_decay)