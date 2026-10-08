import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        # Define the neural network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, grid_size=(8, 8), learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.9, exploration_min=0.01,
                 buffer_size=10000, batch_size=64, target_update_freq=500):
        self.grid_size = grid_size
        self.state_size =  len(grid_size) # dimension of state representation
        self.action_size = 4  # up, right, down, left
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.min_exploration_rate = exploration_min

        #set device on macos
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Build the Q-network
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.step_count = 0
        
    def _state_to_tensor(self, state):
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)
        return torch.FloatTensor(state).to(self.device)
    
    def _state_to_onehot(self, state):
        """Convert (x, y) state to one-hot encoded tensor"""
        if isinstance(state, (tuple, list)):
            x, y = state
        else:
            raise ValueError(f"Expected state as tuple/list, got {type(state)}")
            
        # Create one-hot vector
        onehot = np.zeros(self.state_size, dtype=np.float32)
        # Convert 2D position to 1D index
        index = x * self.grid_size[1] + y
        
        # Safety check
        if 0 <= index < self.state_size:
            onehot[index] = 1.0
        else:
            print(f"Warning: Invalid state {state}, index {index}")
            
        return torch.FloatTensor(onehot).to(self.device)
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)  # Explore: random action
        else:
            # Exploit: use the Q-network to predict the best action
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            # state_tensor = self._state_to_onehot(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor) 
            return torch.argmax(q_values).item()
        
    def update_model(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        # state_tensor = self._state_to_tensor(state)
        # next_state_tensor = self._state_to_tensor(next_state)
        # state_onehot = self._state_to_onehot(state).cpu().numpy()
        # next_state_onehot = self._state_to_onehot(next_state).cpu().numpy()
        
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to learn from
        elif self.step_count % 4 == 0:
            self._replay()
        self.step_count += 1

    def _replay(self):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # state_tensors = []
        # next_state_tensors = []
        # for i in range(len(states)):
        #     state_tensors.append(self._state_to_tensor(states[i]))
        #     next_state_tensors.append(self._state_to_tensor(next_states[i]))
        
        # # Stack tensors into batches
        # states = torch.stack(state_tensors, dim=0)
        # next_states = torch.stack(next_state_tensors, dim=0)

        # Convert other data to tensors
        # convert list to numpy array first
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + ~dones * self.gamma * next_q_values

        # Compute loss and update the Q-network
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_rate_decay)