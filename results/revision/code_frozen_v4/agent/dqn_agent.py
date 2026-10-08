import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, net_arch=None):
        super(QNetwork, self).__init__()
        # Legacy architecture: hidden -> 2*hidden. net_arch=[w1, w2, ...] gives an MLP with
        # those widths (e.g. [256, 256], the rl-baselines3-zoo default for classic control).
        widths = list(net_arch) if net_arch else [hidden_size, 2 * hidden_size]
        dims = [state_dim] + widths
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], action_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)  # No activation on output layer

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
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01,
                 buffer_size=10000, batch_size=64, target_update_freq=5, tau=0.001,
                 hidden_size=128, device=None, net_arch=None, loss='mse', max_grad_norm=1.0,
                 learning_starts=0, train_freq=1, gradient_steps=1,
                 target_update_interval=None, eps_schedule=None):
        """Defaults reproduce the original agent (soft target updates, per-episode ε decay).

        rl-baselines3-zoo style options:
          target_update_interval  hard target copy every N env steps (instead of soft τ updates)
          train_freq / gradient_steps  gradient steps per N env steps (-1 = train_freq steps)
          learning_starts         env steps collected before training
          loss='huber', max_grad_norm, net_arch
          eps_schedule=dict(initial, final, fraction, total_steps)  linear ε per env step;
                                  decay_exploration() then becomes a no-op.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.min_exploration_rate = exploration_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.loss = loss
        self.max_grad_norm = max_grad_norm
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = train_freq if gradient_steps == -1 else gradient_steps
        self.target_update_interval = target_update_interval
        self.eps_schedule = eps_schedule
        self.env_steps = 0
        # FedProx (Li et al., 2020): μ/2 ||θ − θ_glob||² added to the TD loss when prox_ref is set
        self.prox_mu = 0.0
        self.prox_ref = None
        if eps_schedule:
            self.exploration_rate = eps_schedule['initial']

        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )

        # Build Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_size=hidden_size,
                                  net_arch=net_arch).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim, hidden_size=hidden_size,
                                       net_arch=net_arch).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.step_count = 0

    def _state_to_tensor(self, state):
        """Support GridWorld(tuple) 和 Gymnasium(ndarray) """
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, list):
            state = np.array(state, dtype=np.float32)
            
        # 确保是 float32
        if isinstance(state, np.ndarray):
            return torch.FloatTensor(state).to(self.device)
        
        # Fallback
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    # def choose_action(self, state):
    #     if np.random.rand() < self.exploration_rate:
    #         return np.random.randint(self.action_dim)
    #     else:
    #         state_tensor = self._state_to_tensor(state).unsqueeze(0) # Add batch dim
    #         with torch.no_grad():
    #             q_values = self.q_network(state_tensor)
    #         return torch.argmax(q_values).item()

    # softmax action selection
    def choose_action(self, state):
        # state_tensor = self._state_to_tensor(state).unsqueeze(0) # Add batch dim
        # with torch.no_grad():
        #     q_values = self.q_network(state_tensor).squeeze(0)  # Remove batch dim
        #     q_shifted = q_values - torch.max(q_values)  # For numerical stability
        #     exp_q = torch.exp(q_shifted / max(self.exploration_rate, 1e-5))  # Avoid division by zero
        #     action_probs = exp_q / torch.sum(exp_q)
        # action = np.random.choice(self.action_dim, p=action_probs.cpu().numpy())
        # return action
        
        # epsilon-greedy
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)
        else:   
            state_tensor = self._state_to_tensor(state).unsqueeze(0) # Add batch dim
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update_model(self, state, action, reward, next_state, done, next_action=None):
        # Ensure state and next_state are numpy arrays
        if isinstance(state, tuple): state = np.array(state, dtype=np.float32)
        if isinstance(next_state, tuple): next_state = np.array(next_state, dtype=np.float32)

        # Ensure done is bool or float (0/1)
        done_float = 1.0 if done else 0.0
        
        self.replay_buffer.push(state, action, reward, next_state, done_float)
        self.env_steps += 1

        if self.eps_schedule:
            sch = self.eps_schedule
            frac = min(1.0, self.env_steps / max(1.0, sch['fraction'] * sch['total_steps']))
            self.exploration_rate = sch['initial'] + frac * (sch['final'] - sch['initial'])

        if self.target_update_interval and self.env_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if len(self.replay_buffer) < self.batch_size or self.env_steps <= self.learning_starts:
            return
        if self.env_steps % self.train_freq != 0:
            return

        # replay from buffer
        for _ in range(self.gradient_steps):
            self._replay()
            self.step_count += 1

    def _replay(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s, a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: R + gamma * max Q(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        if self.loss == 'huber':
            loss = F.smooth_l1_loss(q_values, target_q_values)
        else:
            loss = F.mse_loss(q_values, target_q_values)

        if self.prox_mu > 0 and self.prox_ref is not None:
            loss = loss + 0.5 * self.prox_mu * sum(
                torch.sum((p - r) ** 2) for p, r in zip(self.q_network.parameters(), self.prox_ref))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # if self.step_count % self.target_update_freq == 0:
        #     self.target_network.load_state_dict(self.q_network.state_dict())

        # Soft update (legacy); zoo mode uses hard copies every target_update_interval env steps
        if not self.target_update_interval:
            self.soft_update_target_network()

    def set_prox_reference(self):
        """Anchor the FedProx term at the current (just-received global) parameters."""
        self.prox_ref = [p.detach().clone() for p in self.q_network.parameters()]

    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    def decay_exploration(self):
        if self.eps_schedule:  # ε follows the per-step linear schedule instead
            return
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_rate_decay)