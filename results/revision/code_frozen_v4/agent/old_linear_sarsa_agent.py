import numpy as np
import random

class LinearSARSAAgent:
    def __init__(self, grid_size, learning_rate=0.01, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01,
                 feature_type='rbf'):  
        """
        Args:
            grid_size: Tuple (height, width) of the grid
            learning_rate: Step size for weight updates
            discount_factor: Gamma
            exploration_rate: Epsilon
            exploration_decay: Epsilon decay rate
            exploration_min: Minimum epsilon
            feature_type: 'tile', 'rbf'
        """
        self.grid_size = grid_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_decay
        self.exploration_min = exploration_min
        self.feature_type = feature_type
        
        # Tile Coding 专用参数配置
        if self.feature_type == 'tile':
            self.num_tilings = 4
            self.tile_grid_res = 4 
        
        # RBF 专用参数 (为了对比，我也帮你修好了 RBF)
        elif self.feature_type == 'rbf':
            self.rbf_sigma = 0.1  
            self.rbf_centers = self._create_rbf_centers()

        self.num_actions = 4
        self.feature_dim = self._get_feature_dimension()
        
        # 初始化权重 (num_actions, feature_dim)
        self.weights = np.zeros((self.num_actions, self.feature_dim))
        
        self.next_action = None
    
    def _get_feature_dimension(self):
        """计算特征向量的长度"""
        if self.feature_type == 'basic':
            return 6
        elif self.feature_type == 'tile':
            # dimension = num_tilings * (tiles_per_row)^2
            return self.num_tilings * (self.tile_grid_res + 1) * (self.tile_grid_res + 1)
        elif self.feature_type == 'rbf':
            return len(self.rbf_centers) + 1  # Centers + Bias
        elif self.feature_type == 'polynomial':
            return 10
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def _get_features(self, state):
        """
        提取特征向量 φ(s)
        """
        x, y = state
        # 归一化到 [0, 1]
        x_norm = x / (self.grid_size[0] - 1)
        y_norm = y / (self.grid_size[1] - 1)
        
        if self.feature_type == 'tile':
            return self._tile_coding_features(x_norm, y_norm)
        
        elif self.feature_type == 'basic':
            return np.array([x_norm, y_norm, x_norm**2, y_norm**2, x_norm*y_norm, 1.0])
        
        elif self.feature_type == 'rbf':
            return self._rbf_features(x_norm, y_norm)
            
        elif self.feature_type == 'polynomial':
            return np.array([1.0, x_norm, y_norm, x_norm**2, y_norm**2, x_norm*y_norm, 
                           x_norm**3, y_norm**3, x_norm**2*y_norm, x_norm*y_norm**2])
        return np.zeros(self.feature_dim)

    def _tile_coding_features(self, x, y):
        """
        核心代码：Tile Coding 实现
        输入 x, y 已经在 [0, 1] 范围内
        """
        features = np.zeros(self.feature_dim)
        
        tiles_per_row = self.tile_grid_res + 1
        
        for tiling_idx in range(self.num_tilings):
            # Offset
            # i / num_tilings * tile_width
            offset_x = tiling_idx * (1.0 / (self.num_tilings * self.tile_grid_res))
            offset_y = tiling_idx * (1.0 / (self.num_tilings * self.tile_grid_res))

            # Tile Coordinates
            # x (0~1) * res (4) -> 0~4.xxx
            tile_x = int((x + offset_x) * self.tile_grid_res)
            tile_y = int((y + offset_y) * self.tile_grid_res)
            
            # 3. calculate the starting index for this tiling layer
            layer_start_idx = tiling_idx * (tiles_per_row * tiles_per_row)

            # 4. calculate the local index for the current tile within the layer
            local_idx = tile_x * tiles_per_row + tile_y
            
            # 5. set the corresponding feature to 1
            final_idx = layer_start_idx + local_idx
            if final_idx < self.feature_dim:
                features[final_idx] = 1.0
        
        return features

    def _create_rbf_centers(self):
        """RBF Centers: 5x5 grid in [0,1]x[0,1]"""
        centers = []
        for i in range(5):
            for j in range(5):
                centers.append([i / 4.0, j / 4.0])
        return np.array(centers)

    def _rbf_features(self, x, y):
        features = np.zeros(self.feature_dim)
        point = np.array([x, y])
        for i, center in enumerate(self.rbf_centers):
            distance_sq = np.sum((point - center) ** 2)
            features[i] = np.exp(-distance_sq / (2 * self.rbf_sigma ** 2))
        features[-1] = 1.0
        return features
    
    def get_q_value(self, state, action):
        features = self._get_features(state)
        return np.dot(self.weights[action], features)
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
            max_q = np.max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_model(self, state, action, reward, next_state, done):
        features = self._get_features(state)
        current_q = self.get_q_value(state, action)
        
        if done:
            target = reward
        else:
            # SARSA Update
            if self.next_action is None:
                self.next_action = self.choose_action(next_state)
            next_q = self.get_q_value(next_state, self.next_action)
            target = reward + self.gamma * next_q
        
        td_error = target - current_q
        
        # w += alpha * error * features
        self.weights[action] += self.lr * td_error * features
        
        if done:
            self.next_action = None
            
    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_min, 
                                   self.exploration_rate * self.exploration_rate_decay)