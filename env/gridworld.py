import numpy as np
import random
GRID_SEED = 42

# Define the gridworld environment
class GridWorld:
    def __init__(self, size=(8, 8), grid_seed=GRID_SEED):
        # self.grid = np.zeros(size)
        # # Define one goal and 30% obstacles
        # if grid_seed is not None:
        #     random.seed(grid_seed)
        #     np.random.seed(grid_seed)

        # Generate goal position
        # goal_position = (random.randint(0, size[0]-1), random.randint(0, size[1]-1))
        # self.grid[goal_position] = 1  # Goal
        # self.goal_position = goal_position
        
        # # Generate obstacles (30% of grid)
        # num_obstacles = int(size[0] * size[1] * 0.3)
        # obstacles = set()
        # while len(obstacles) < num_obstacles:
        #     pos = (random.randint(0, size[0]-1), random.randint(0, size[1]-1))
        #     if pos != goal_position:
        #         obstacles.add(pos)
        
        # for obs in obstacles:
        #     self.grid[obs] = -1  # Obstacle
        
        #  # CRITICAL: Reset random seed after grid generation
        # random.seed(None)
        # np.random.seed(None)
        self.grid = np.array([
            # 8x8 grid world with strategic obstacles and goal at bottom-right
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, -1, -1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.state = self.reset()
        self.max_steps = 50

    def reset(self):
        # randomly place the agent in a valid starting position (not on an obstacle or goal)
        valid_positions = [(i, j) for i in range(self.grid.shape[0]) 
                           for j in range(self.grid.shape[1]) 
                           if self.grid[i, j] == 0]
        self.state = random.choice(valid_positions)
        self.steps = 0
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:  # Move up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Move right
            next_state[1] = min(self.grid.shape[1]-1, state[1] + 1)
        elif action == 2:  # Move down
            next_state[0] = min(self.grid.shape[0]-1, state[0] + 1)
        elif action == 3:  # Move left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)
    def is_success(self, state):
        return self.grid[state] == 1

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        if self.grid[next_state] == 1:  # Reached goal
            reward = 50  # ✅ Positive reward for reaching the goal
            done = True
        elif self.grid[next_state] == -1:  # Hit obstacle
            reward = -1 # ✅ Negative reward for hitting an obstacle
            done = False
            next_state = self.state  # Stay in the same state
        else:  # Normal step
            reward = -1  # ✅negative reward for each step to encourage efficiency
            done = False

        self.state = next_state
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        return next_state, reward, done
    
    def render(self):
        for i in range(self.grid.shape[0]):
            row = ""
            for j in range(self.grid.shape[1]):
                if (i, j) == self.state:
                    row += " A "  # Agent's current position
                elif self.grid[i, j] == 1:
                    row += " G "  # Goal
                elif self.grid[i, j] == -1:
                    row += " X "  # Obstacle
                else:
                    row += " . "  # Empty cell
            print(row)
        print(f"steps: {self.steps}")
        print()
