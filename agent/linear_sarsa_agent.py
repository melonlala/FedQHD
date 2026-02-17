import numpy as np
import random
from env.bounds import state_bounds

class LinearSARSAAgent:
    def __init__(self, state_dim=8, action_dim=4, 
                 learning_rate=0.001, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01,
                 feature_type='rbf', feature_dim=1000, state_bounds=None, random_seed=24): 
        """
        Args:
            state_dim: LunarLander 为 8
            action_dim: LunarLander 为 4
            feature_dim: RBF 特征的数量 (即中心点的数量)
            feature_type: 'rbf'
        """
        self.state_dim = state_dim
        self.num_actions = action_dim
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_min = exploration_min
        self.feature_type = feature_type
        
        # --- RBF 专用配置 ---
        if self.feature_type == 'rbf':
            self.num_centers = feature_dim
            self.feature_dim = self.num_centers + 1 # +1 是 Bias (偏置项)
            
            # Sigma (带宽): 决定了每个中心点负责的范围。
            # 因为我们做了归一化(0到1)，0.5 是一个经验值，表示覆盖半径约为半个空间
            self.rbf_sigma = 0.5 
            
            # 初始化中心点
            random.seed(random_seed)
            self.rbf_centers = self._create_rbf_centers()
            
        else:
            raise ValueError("This code is optimized for 'rbf'.")

        # 初始化权重 (Actions, Features)
        self.weights = np.zeros((self.num_actions, self.feature_dim))
        
        # SARSA 专用变量
        self.next_action = None
        
        if state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
        else:
            self.state_bounds = np.array([[-1, 1]] * self.state_dim)

    def _normalize_state(self, state):
        """
        normalize state to [0, 1] based on predefined bounds
        clip values outside the bounds
        """
        norm_state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            min_val, max_val = self.state_bounds[i]
            denom = (max_val - min_val)
            if denom == 0:
                norm_state[i] = 0.0
                continue
            # scale to [-1, 1]
            norm_state[i] = 2 * (state[i] - min_val) / denom - 1
        # Replace non-finite values and clip to valid range
        norm_state = np.nan_to_num(norm_state, nan=0.0, posinf=0.0, neginf=0.0)
        norm_state = np.clip(norm_state, -1.0, 1.0)
        return norm_state

    def _create_rbf_centers(self):
        """
        randomly generate RBF centers within [-1, 1] range for each dimension
        """
        return np.random.uniform(-1, 1, (self.num_centers, self.state_dim))

    def _get_features(self, state):
        """
        计算 RBF 特征向量: φ(s)
        φ_i(s) = exp( - ||s - c_i||^2 / (2 * sigma^2) )
        """
        # 1. 先归一化状态
        norm_state = self._normalize_state(state)
        
        features = np.zeros(self.feature_dim)
        
        # 2. 向量化计算距离 (比 for 循环快)
        # diff shape: (num_centers, 8)
        diff = norm_state - self.rbf_centers
        
        # squared_distances shape: (num_centers,)
        dist_sq = np.sum(diff**2, axis=1)
        
        # 3. 计算高斯激活值
        features[:self.num_centers] = np.exp(-dist_sq / (2 * self.rbf_sigma**2))
        
        # 4. 最后一个特征设为 Bias (常数 1)
        features[-1] = 1.0
        
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_q_value(self, state, action):
        features = self._get_features(state)
        return np.dot(self.weights[action], features)
    
    # def choose_action(self, state):
    #     if random.uniform(0, 1) < self.exploration_rate:
    #         return random.randint(0, self.num_actions - 1)
    #     else:
    #         q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
    #         max_q = np.max(q_values)
    #         # 随机打破平局
    #         best_actions = [i for i, q in enumerate(q_values) if q == max_q]
    #         return random.choice(best_actions)

    def choose_action(self, state):
        # Epsilon-greedy action selection (fixed from softmax)
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action
            q_values = np.array([self.get_q_value(state, a) for a in range(self.num_actions)], dtype=float)
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
        """Standard SARSA Update"""
        features = self._get_features(state)
        current_q = self.get_q_value(state, action)
        
        if done:
            target = reward
        else:
            if self.next_action is None:
                self.next_action = self.choose_action(next_state)
            next_q = self.get_q_value(next_state, self.next_action)
            target = reward + self.gamma * next_q
        
        td_error = target - current_q
        if not np.isfinite(td_error) or not np.all(np.isfinite(features)):
            if done:
                self.next_action = None
            return
        
        self.weights[action] += self.lr * td_error * features
        
        if done:
            self.next_action = None
            
    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_min, 
                                   self.exploration_rate * self.exploration_rate_decay)