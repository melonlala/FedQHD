import numpy as np
import random

class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) Agent
    An on-policy TD control algorithm that updates Q-values based on the action actually taken.
    """
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
        # Initialize Q-table with zeros for grid_size with 4 actions (up, right, down, left)
        self.q_table = np.zeros((grid_size[0], grid_size[1], 4))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Store the next action for SARSA update
        self.next_action = None
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        This is the same as Q-learning since both use epsilon-greedy for action selection.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore: random action
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit: best action
    
    def update_model(self, state, action, reward, next_state, done):
        """
        SARSA update rule: Q(s,a) <- Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]
        
        Key difference from Q-learning:
        - Q-learning uses: max Q(s',a') (off-policy, uses best possible action)
        - SARSA uses: Q(s',a') where a' is the actual next action taken (on-policy)
        """
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Choose the next action using current policy
            if self.next_action is None:
                self.next_action = self.choose_action(next_state)
            
            # Use Q-value of the actual next action (SARSA's key feature)
            next_q = self.q_table[next_state[0], next_state[1], self.next_action]
            target = reward + self.gamma * next_q
        
        # Update Q-value
        current_q = self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] = current_q + self.lr * (target - current_q)
        
        # Reset next_action for next episode if done
        if done:
            self.next_action = None
    
    def decay_exploration(self):
        """Decay exploration rate after each episode"""
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_rate_decay)
    