# mini_pacman_rl.py

import random
import numpy as np
from PIL import Image, ImageDraw



def generate_maze_map(width, height, coin_prob=0.10):
    """
    Labyrinthe avec :
    - bordures externes en '#'
    - couloirs internes
    - pièces 'C'
    - 'P' et 'F' placés ensuite (dans l'env)
    """
    if width < 5 or height < 5:
        raise ValueError("La taille minimale recommandée est au moins 5x5.")

    grid = [["#" for _ in range(width)] for _ in range(height)]

    cells = []
    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            grid[y][x] = "."
            cells.append((x, y))

    if not cells:
        raise ValueError("Dimensions trop petites pour construire un labyrinthe interne.")

    start_x, start_y = random.choice(cells)
    visited = set()
    stack = [(start_x, start_y)]
    visited.add((start_x, start_y))

    directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    while stack:
        x, y = stack[-1]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if (nx, ny) not in visited and grid[ny][nx] == ".":
                    neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            mx = (x + nx) // 2
            my = (y + ny) // 2
            grid[my][mx] = "."
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if x == 1 or x == width - 2 or y == 1 or y == height - 2:
                if grid[y][x] == "#":
                    grid[y][x] = "."

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == "." and random.random() < coin_prob:
                grid[y][x] = "C"

    return ["".join(row) for row in grid]




class MiniPacmanEnv:
    ACTIONS = {
        0: (0, -1),  
        1: (0, 1),    
        2: (-1, 0),  
        3: (1, 0)     
    }

    def __init__(self, width=15, height=11, max_steps=100, coin_prob=0.15):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.coin_prob = coin_prob

        self._build_from_map(generate_maze_map(width, height, coin_prob))


    def render_image(self, cell_size=32):
        width_px = self.width * cell_size
        height_px = self.height * cell_size

        img = Image.new("RGB", (width_px, height_px), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # couleurs
        COLOR_BG = (0, 0, 0)
        COLOR_WALL = (80, 80, 80)
        COLOR_GRID = (60, 60, 160)
        COLOR_COIN = (255, 255, 0)
        COLOR_PACMAN = (255, 255, 0)
        COLOR_GHOST = (255, 0, 0)

        for y in range(self.height):
            for x in range(self.width):
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                if (x, y) in self.walls:
                    draw.rectangle([x0, y0, x1, y1], fill=COLOR_WALL)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=COLOR_BG, outline=COLOR_GRID)

        for pos, idx in self.coin_index.items():
            if (self.coins_mask >> idx) & 1:
                x, y = pos
                cx = x * cell_size + cell_size // 2
                cy = y * cell_size + cell_size // 2
                r = cell_size // 6
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOR_COIN)

        px, py = self.pacman_pos
        cx = px * cell_size + cell_size // 2
        cy = py * cell_size + cell_size // 2
        r = cell_size // 2 - 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOR_PACMAN)

        fx, fy = self.ghost_pos
        x0 = fx * cell_size + 4
        y0 = fy * cell_size + 4
        x1 = (fx + 1) * cell_size - 4
        y1 = (fy + 1) * cell_size - 4
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_GHOST)

        return img

    def _build_from_map(self, grid_str):
        self.grid_str = grid_str
        self.walls = set()
        self.coins_positions = []
        self.pacman_start = None
        self.ghost_start = None

        for y in range(self.height):
            for x in range(self.width):
                c = grid_str[y][x]
                if c == "#":
                    self.walls.add((x, y))
                elif c == "C":
                    self.coins_positions.append((x, y))
                elif c == "P":
                    self.pacman_start = (x, y)
                elif c == "F":
                    self.ghost_start = (x, y)

        free_cells = [(x, y) for y in range(1, self.height - 1)
                      for x in range(1, self.width - 1)
                      if (x, y) not in self.walls]

        if self.pacman_start is None:
            self.pacman_start = random.choice(free_cells)
        if self.ghost_start is None:
            choice = random.choice(free_cells)
            while choice == self.pacman_start:
                choice = random.choice(free_cells)
            self.ghost_start = choice

        self.coin_index = {pos: i for i, pos in enumerate(self.coins_positions)}
        self.n_coins = len(self.coins_positions)

    def reset(self, regenerate_maze=False):
        if regenerate_maze:
            self._build_from_map(generate_maze_map(self.width, self.height, self.coin_prob))

        self.pacman_pos = self.pacman_start
        self.ghost_pos = self.ghost_start
        self.coins_mask = (1 << self.n_coins) - 1
        self.steps = 0
        self.visit_counts = {}
        return self._get_state()

    def _get_state(self):
        px, py = self.pacman_pos
        fx, fy = self.ghost_pos
        return (px, py, fx, fy, self.coins_mask)

    def _valid(self, x, y):
        return (0 <= x < self.width) and (0 <= y < self.height) and ((x, y) not in self.walls)

    def _move_ghost_random(self):
        x, y = self.ghost_pos
        candidates = []
        for (dx, dy) in MiniPacmanEnv.ACTIONS.values():
            nx, ny = x + dx, y + dy
            if self._valid(nx, ny):
                candidates.append((nx, ny))
        candidates.append((x, y))
        self.ghost_pos = random.choice(candidates)

    def step(self, action):
        self.steps += 1
        reward = -0.1
        done = False

        old_px, old_py = self.pacman_pos
        dx, dy = MiniPacmanEnv.ACTIONS[action]
        new_px = old_px + dx
        new_py = old_py + dy

        if self._valid(new_px, new_py):
            self.pacman_pos = (new_px, new_py)
        else:
            self.pacman_pos = (old_px, old_py)
            reward -= 0.5 

        if self.pacman_pos == (old_px, old_py):
            reward -= 0.2  

        if self.pacman_pos in self.coin_index:
            idx = self.coin_index[self.pacman_pos]
            if (self.coins_mask >> idx) & 1:
                self.coins_mask &= ~(1 << idx)
                reward += 5.0

        if self.coins_mask == 0:
            reward += 20.0
            done = True

        self._move_ghost_random()

        if self.pacman_pos == self.ghost_pos:
            reward -= 20.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        state = self._get_state()
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1
        reward -= 0.01 * (self.visit_counts[state] - 1)

        return state, reward, done, {}

    def get_ascii_grid(self):
        """
        Retourne la grille actuelle sous forme de liste de strings
        avec P, F, C, #, .
        """
        grid = [list(row) for row in self.grid_str]

        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] in ("P", "F"):
                    grid[y][x] = "."

        for pos, idx in self.coin_index.items():
            x, y = pos
            if (self.coins_mask >> idx) & 1:
                grid[y][x] = "C"
            else:
                if grid[y][x] == "C":
                    grid[y][x] = "."

        px, py = self.pacman_pos
        fx, fy = self.ghost_pos
        grid[py][px] = "P"
        grid[fy][fx] = "F"

        return ["".join(row) for row in grid]

class QLearningAgent:
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions, dtype=float)
        return self.Q[state]

    def choose_action(self, state):
        q_values = self.get_Q(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        q_values = self.get_Q(state)
        target = reward
        if not done:
            next_q = self.get_Q(next_state)
            target += self.gamma * np.max(next_q)
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



def train_agent(env, agent, n_episodes=5000, max_steps=10000, regenerate_maze=False):
    rewards_per_episode = []
    coins_per_episode = []

    for ep in range(n_episodes):
        state = env.reset(regenerate_maze=regenerate_maze)
        total_reward = 0.0
        coins_start = bin(env.coins_mask).count("1")

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        coins_end = bin(env.coins_mask).count("1")
        collected = coins_start - coins_end

        rewards_per_episode.append(total_reward)
        coins_per_episode.append(collected)
        agent.decay_epsilon()

    return rewards_per_episode, coins_per_episode


def play_episode(env, agent, max_steps=100):
    """
    Joue une partie en exploitation pure (epsilon=0) et
    retourne une liste de grilles ASCII (pour animation dans Streamlit).
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    frames = []
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        frames.append(env.get_ascii_grid())
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        steps += 1

    frames.append(env.get_ascii_grid())

    agent.epsilon = old_epsilon
    return frames

def play_episode_images(env, agent, max_steps=100, cell_size=32, epsilon_demo=0.05):
    old_epsilon = agent.epsilon
    agent.epsilon = epsilon_demo  

    frames = []
    state = env.reset()  
    done = False
    steps = 0

    frames.append(env.render_image(cell_size=cell_size))

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        steps += 1
        frames.append(env.render_image(cell_size=cell_size))

    agent.epsilon = old_epsilon
    return frames
