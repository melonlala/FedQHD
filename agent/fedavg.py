import numpy as np
from agent.q_learning_agent import QLearningAgent
from agent.dqn_agent import DQNAgent, QNetwork
from agent.qhd_agent import QHDAgent
from agent.sarsa_agent import SARSAAgent
from agent.linear_sarsa_agent import LinearSARSAAgent
from agent.linear_sarsa_agent_structured import LinearSARSAAgentStructured
from agent.hd_sarsa_agent import HDSARSAAgent
import torch
import random
AGENT_DICT = {
    'q_learning': QLearningAgent,
    'dqn': DQNAgent,
    'qhd': QHDAgent,
    'sarsa': SARSAAgent,
    'linear_sarsa': LinearSARSAAgent,
    'hd_sarsa': HDSARSAAgent
}


class FedAvgAgent:
    """
    Federated Averaging coordinator for multiple Q-learning agents.
    Manages multiple agents that learn independently and periodically aggregate their knowledge.
    """
    def __init__(self, state_dim, action_dim,  agent_type, num_agents, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01, hd_dim=None, state_bounds=None, use_structured_rbf=False, num_rbf_centers=1000):
        self.num_agents = num_agents
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_state = 42  # For reproducibility

        if agent_type == 'dqn':
            self.agents = [DQNAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate, discount_factor=discount_factor,
                                    exploration_rate=exploration_rate, exploration_decay=exploration_decay, batch_size=64, target_update_freq=100)  
                           for _ in range(num_agents)]
            self.global_model = QNetwork(state_dim, action_dim)
        elif agent_type == 'q_learning':
            self.agents = [QLearningAgent(state_space=(12, 4), action_dim=action_dim, learning_rate=learning_rate, discount_factor=discount_factor, 
                                         exploration_rate=exploration_rate, exploration_decay=exploration_decay, exploration_min=exploration_min, state_bounds=state_bounds) 
                           for _ in range(num_agents)]
            self.global_model = np.zeros((12*4, action_dim))
        elif agent_type == 'qhd':
            self.agents = [QHDAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate, discount_factor=discount_factor, 
                                   exploration_rate=exploration_rate, exploration_decay=exploration_decay, exploration_min=exploration_min, 
                                   random_seed=self.random_state, state_bounds=state_bounds) 
                           for _ in range(num_agents)]
            # self.global_model = np.zeros((action_dim, 10000), dtype=np.float64)

            # randomly initialize and normalize
            self.global_model = np.zeros((action_dim, 10000), dtype=np.float64)
            for i in range(action_dim):
                self.global_model[i] = np.random.randn(10000)
                self.global_model[i] /= np.linalg.norm(self.global_model[i])

        elif agent_type == 'sarsa':
            self.agents = [SARSAAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate, discount_factor=discount_factor, 
                                     exploration_rate=exploration_rate, exploration_decay=exploration_decay, exploration_min=exploration_min, state_bounds=state_bounds) 
                           for _ in range(num_agents)]
            self.global_model = np.zeros((state_dim[0], state_dim[1], action_dim))
        elif agent_type == 'linear_sarsa':
            # Choose between structured RBF (64 centers) or random RBF (1000 centers)
            if use_structured_rbf:
                self.agents = [LinearSARSAAgentStructured(state_dim=state_dim, action_dim=action_dim,
                                                         learning_rate=learning_rate, discount_factor=discount_factor,
                                                         exploration_rate=exploration_rate, exploration_decay=exploration_decay,
                                                         exploration_min=exploration_min, num_rbf_per_dim=8,
                                                         state_bounds=state_bounds)
                               for _ in range(num_agents)]
            else:
                self.agents = [LinearSARSAAgent(state_dim=state_dim, action_dim=action_dim,
                                               learning_rate=learning_rate, discount_factor=discount_factor,
                                               exploration_rate=exploration_rate, exploration_decay=exploration_decay,
                                               exploration_min=exploration_min, feature_dim=num_rbf_centers,
                                               state_bounds=state_bounds)
                               for _ in range(num_agents)]
            self.global_model = np.zeros((action_dim, self.agents[0].feature_dim))
        elif agent_type == 'hd_sarsa':
            self.agents = [HDSARSAAgent(state_dim=state_dim, action_dim=action_dim, hd_dim=hd_dim,
                                       learning_rate=learning_rate, discount_factor=discount_factor,
                                       exploration_rate=exploration_rate, exploration_decay=exploration_decay,
                                       exploration_min=exploration_min,
                                       random_seed=self.random_state, state_bounds=state_bounds)
                           for _ in range(num_agents)]
            self.global_model = np.zeros((action_dim, 10000), dtype=np.float64)

        self.agent_type = agent_type
        
    def get_agent(self, agent_id):
        """Get a specific agent by ID"""
        return self.agents[agent_id]
    
    
    def aggregate(self):
        """
        FedAvg: Average all agents' local models to create a global model.
        This is the core of Federated Averaging.
        """
        # Sum all local models
        if self.agent_type == 'dqn':
            first_agent_state = self.agents[0].q_network.state_dict()
            # Initialize aggregated parameters
            aggregated_state = {}
            # For each parameter in the model
            for key in first_agent_state.keys():
                # Stack all agents' parameters for this key
                param_list = []
                for agent in self.agents:
                    param_list.append(agent.q_network.state_dict()[key])
                
                # Stack tensors and compute mean
                stacked_params = torch.stack(param_list)
                aggregated_state[key] = torch.mean(stacked_params, dim=0)
            
            # Update global model
            self.global_model.load_state_dict(aggregated_state)
            return aggregated_state

        elif self.agent_type in ['q_learning', 'sarsa']:
            sum_local_models = np.zeros_like(self.agents[0].q_table)
            for agent in self.agents:
                sum_local_models += agent.q_table
            self.global_model = sum_local_models / self.num_agents
        elif self.agent_type in ['qhd', 'hd_sarsa']:
            sum_local_models = np.zeros_like(self.agents[0].model_vectors, dtype=np.longdouble)
            for agent in self.agents:
                sum_local_models += agent.model_vectors.astype(np.longdouble)
            averaged = sum_local_models / np.longdouble(self.num_agents)
            self.global_model = averaged.astype(np.float64)
        elif self.agent_type in ['linear_sarsa']:
            sum_local_models = np.zeros_like(self.agents[0].weights)
            for agent in self.agents:
                sum_local_models += agent.weights
            self.global_model = sum_local_models / self.num_agents

        return self.global_model

    def distribute(self):
        """
        Distribute the global model back to all agents.
        Each agent starts the next round with the aggregated knowledge.
        """
        for agent in self.agents:
            if self.agent_type == 'dqn':
                agent.q_network.load_state_dict(self.global_model.state_dict())
            elif self.agent_type in ['q_learning', 'sarsa']:
                agent.q_table = self.global_model.copy()
            elif self.agent_type in ['qhd', 'hd_sarsa']:
                agent.model_vectors = np.copy(self.global_model)
            elif self.agent_type in ['linear_sarsa']:
                agent.weights = self.global_model.copy()

    def decay_exploration(self):
        """Decay exploration rate for global model"""
        for agent in self.agents:
            agent.decay_exploration()

    def get_best_agent_model(self):
        """Return the global model for testing"""
        return self.global_model.state_dict() if self.agent_type == 'dqn' else self.global_model