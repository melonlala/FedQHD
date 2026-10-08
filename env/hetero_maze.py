"""HeteroMaze: a four-rooms navigation task whose layout differs across federated clients.

The maze is an 11x11 grid of cells. Walls split it into four rooms. Each of the four wall
segments has one doorway, and the agent starts in the top-left room and must reach a goal
cell in the bottom-right room. Cells are 1x1, so the position s = (x, y) lies in
[0, 11) x [0, 11), with x the column and y the row.

    level 0 (every client)         level 1, u_i = -0.5            level 1, u_i = +1
    . . . . . # . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    . . . . . # . . . . .          . . . . . . . . . . .          . . . . . # . . . . .
    . . . . . . . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    . . . . . # . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    . . . . . # . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    # # . # # # # # . # #          # # # . # # # . # # #          . # # # # # # # # # .
    . . . . . # . . . . .          . . . . . # . . . . .          . . . . . . . . . . .
    . . . . . # . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    . . . . . . . . . . .          . . . . . # . . . . .          . . . . . # . . . . .
    . . . . . # . . . . .          . . . . . . . . . . .          . . . . . # . . . . .
    . . . . . # . . . . G          . . . . . # . . . . G          . . . . . # . . . . G

Dynamics: action a in {up, right, down, left} moves the agent by ``step`` cells (default 1) plus
Gaussian noise (std ``noise``) on each axis. A move along an axis is cancelled if its path would
touch a wall cell or leave the grid (per-axis, so the agent slides along walls). With
``step=1, noise=0`` and a cell-centred start this reduces to a deterministic grid world.

Reward: -1 per step. The episode terminates when the agent enters the goal cell and is
truncated after ``max_steps`` steps, so the return is -(number of steps) and lies in
[-max_steps, -1], as in MountainCar.

Heterogeneity (client-specific MDPs): client i gets a factor u_i in [-1, 1], evenly spaced
over the N clients (0 for the server copy). With ``mode`` containing
  * 'layout': every doorway is shifted along its wall segment by round(2 * level * u_i)
    cells (alternating directions across segments, clipped to the segment), and for
    |u_i| * level >= 0.75 one exit of the start room is closed (u_i > 0: the right exit,
    u_i < 0: the lower exit), so these clients must leave the start room by different
    routes. This changes the transition dynamics and hence Q*;
  * 'goal':   the goal cell moves along the bottom row of the goal room by
    round(2 * level * (1 + u_i)) cells. This changes the reward function.
``level = 0`` gives identical MDPs for all clients.
"""

from collections import deque

import gymnasium as gym
import numpy as np

SIZE = 11                       # cells per side
WALL = 5                        # index of the central wall row / column
SEGMENT_LEN = 5                 # cells per wall segment (between border and centre)
BASE_DOOR = 2                   # doorway index within each segment at level 0
DOOR_SIGNS = (+1, -1, -1, +1)   # shift direction per segment: top, bottom, left, right
CLOSE_AT = 0.75                 # |u_i| * level at which one start-room exit is closed
BASE_GOAL = (10, 10)            # (row, col) of the goal cell at level 0
ACTIONS = np.array([[0.0, -1.0],   # up    (dx, dy)
                    [1.0, 0.0],    # right
                    [0.0, 1.0],    # down
                    [-1.0, 0.0]])  # left


def _round_half_away(v: float) -> int:
    return int(np.sign(v) * np.floor(abs(v) + 0.5))


def client_factor(client_id, num_clients: int) -> float:
    """u_i in [-1, 1], evenly spaced over the clients; 0 for the server (client_id None)."""
    if client_id is None or num_clients <= 1:
        return 0.0
    return -1.0 + 2.0 * (client_id % num_clients) / (num_clients - 1)


def make_layout(level: float = 0.0, u: float = 0.0, mode: str = 'layout'):
    """Return (walls, goal, doors).

    walls: (SIZE, SIZE) bool array, True for wall cells (indexed [row, col]);
    goal:  (row, col) of the goal cell;
    doors: doorway index (0..SEGMENT_LEN-1) of the top, bottom, left and right segments.
    """
    use_layout = 'layout' in mode
    use_goal = 'goal' in mode
    doors = []
    for sign in DOOR_SIGNS:
        shift = _round_half_away(2.0 * level * u * sign) if use_layout else 0
        doors.append(int(np.clip(BASE_DOOR + shift, 0, SEGMENT_LEN - 1)))
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[:, WALL] = True
    walls[WALL, :] = True
    top, bottom, left, right = doors
    # Strong layout heterogeneity: for |u| * level >= CLOSE_AT one exit of the start room is
    # closed, so the optimal route out of it differs between clients (u > 0: only the lower
    # exit is open, u < 0: only the right exit is open).
    close_top = use_layout and u * level >= CLOSE_AT
    close_left = use_layout and -u * level >= CLOSE_AT
    if not close_top:
        walls[top, WALL] = False                 # vertical wall, upper segment (rows 0-4)
    walls[WALL + 1 + bottom, WALL] = False       # vertical wall, lower segment (rows 6-10)
    if not close_left:
        walls[WALL, left] = False                # horizontal wall, left segment (cols 0-4)
    walls[WALL, WALL + 1 + right] = False        # horizontal wall, right segment (cols 6-10)
    if close_top:
        doors[0] = -1
    if close_left:
        doors[2] = -1
    goal_shift = _round_half_away(2.0 * level * (1.0 + u)) if use_goal else 0
    goal = (BASE_GOAL[0], int(np.clip(BASE_GOAL[1] - goal_shift, WALL + 1, SIZE - 1)))
    return walls, goal, tuple(doors)


def shortest_path_lengths(walls: np.ndarray, goal) -> np.ndarray:
    """Grid (4-neighbour) shortest-path length from every free cell to the goal (inf if
    unreachable). Used to check connectivity and to bound the optimal return."""
    dist = np.full(walls.shape, np.inf)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and not walls[nr, nc] and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


class HeteroMaze(gym.Env):
    """Four-rooms maze with a continuous (x, y) observation; see the module docstring."""

    metadata = {'render_modes': []}

    def __init__(self, level: float = 0.0, u: float = 0.0, mode: str = 'layout',
                 step: float = 1.0, noise: float = 0.1, max_steps: int = 200,
                 start: str = 'room'):
        super().__init__()
        if start not in ('room', 'anywhere'):
            raise ValueError(f"start must be 'room' or 'anywhere', got {start!r}")
        self.walls, self.goal, self.doors = make_layout(level, u, mode)
        self.step_size = float(step)
        self.noise = float(noise)
        self.max_steps = int(max_steps)
        self.start = start
        self.observation_space = gym.spaces.Box(0.0, float(SIZE), shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        free = ~self.walls
        free[self.goal] = False
        if start == 'room':
            room = np.zeros_like(free)
            room[:WALL, :WALL] = True
            free &= room
        self._start_cells = np.argwhere(free)          # (k, 2) of (row, col)
        self.pos = np.zeros(2)
        self.steps = 0
        self.reached_goal = False

    def _cell(self, pos):
        return int(pos[1]), int(pos[0])                 # (row, col)

    def _blocked(self, pos) -> bool:
        if not (0.0 <= pos[0] < SIZE and 0.0 <= pos[1] < SIZE):
            return True
        return bool(self.walls[self._cell(pos)])

    def _path_blocked(self, start, end, axis: int) -> bool:
        """True if an axis-aligned move from start to end leaves the grid or touches a wall
        cell anywhere along the way (not only at the end point: a move longer than one cell
        could otherwise jump over a one-cell wall)."""
        if self._blocked(end):
            return True
        lo, hi = sorted((int(np.floor(start[axis])), int(np.floor(end[axis]))))
        other = int(start[1 - axis])
        for k in range(lo, hi + 1):
            cell = (other, k) if axis == 0 else (k, other)   # (row, col)
            if self.walls[cell]:
                return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r, c = self._start_cells[self.np_random.integers(len(self._start_cells))]
        if self.step_size == 1.0 and self.noise == 0.0:
            offset = np.array([0.5, 0.5])               # grid-world variant: cell centres
        else:
            offset = self.np_random.uniform(0.05, 0.95, size=2)
        self.pos = np.array([c, r], dtype=float) + offset
        self.steps = 0
        self.reached_goal = False
        return self.pos.astype(np.float32), {}

    def step(self, action):
        move = ACTIONS[int(action)] * self.step_size
        if self.noise > 0:
            move = move + self.np_random.normal(0.0, self.noise, size=2)
        for axis in (0, 1):                             # per-axis collision: slide along walls
            trial = self.pos.copy()
            trial[axis] += move[axis]
            if not self._path_blocked(self.pos, trial, axis):
                self.pos = trial
        self.steps += 1
        terminated = self._cell(self.pos) == self.goal
        self.reached_goal = bool(terminated)
        truncated = (not terminated) and self.steps >= self.max_steps
        return self.pos.astype(np.float32), -1.0, bool(terminated), bool(truncated), {}

    def render_ascii(self) -> str:
        rows = []
        for r in range(SIZE):
            row = []
            for c in range(SIZE):
                if (r, c) == self._cell(self.pos):
                    row.append('A')
                elif (r, c) == self.goal:
                    row.append('G')
                else:
                    row.append('#' if self.walls[r, c] else '.')
            rows.append(' '.join(row))
        return '\n'.join(rows)


class HeteroMazeWrapper:
    """Same interface as the other wrappers in env/utils.py."""

    def __init__(self, **maze_kwargs):
        self.env = HeteroMaze(**maze_kwargs)
        self.state_dim = 2
        self.action_dim = 4

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated or truncated, truncated, info

    def is_success(self, state=None):
        return self.env.reached_goal

    def close(self):
        self.env.close()


def make_maze_env(args, client_id=None):
    """Build client ``client_id``'s maze from the run arguments.

    Uses ``--env_hetero {layout, goal, layout+goal}`` and ``--env_hetero_level``; any other
    ``--env_hetero`` value (e.g. 'none') gives the level-0 maze for every client. Returns
    (wrapper, description dict).
    """
    mode = getattr(args, 'env_hetero', 'none') or 'none'
    level = float(getattr(args, 'env_hetero_level', 0.0) or 0.0)
    if mode not in ('layout', 'goal', 'layout+goal'):
        mode, level = 'layout', 0.0
    u = client_factor(client_id, int(getattr(args, 'agent_num', 1)))
    wrapper = HeteroMazeWrapper(
        level=level, u=u, mode=mode,
        step=float(getattr(args, 'maze_step', 1.0)),
        noise=float(getattr(args, 'maze_noise', 0.1)),
        max_steps=int(getattr(args, 'max_episode_steps', None) or 200),
        start=getattr(args, 'maze_start', 'room'),
    )
    maze = wrapper.env
    desc = {'maze_doors': list(maze.doors), 'maze_goal': list(maze.goal), 'maze_u': u,
            'maze_level': level, 'maze_mode': mode}
    return wrapper, desc
