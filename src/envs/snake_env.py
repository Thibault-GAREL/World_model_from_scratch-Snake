"""Headless Snake environment (Gym-like) for the JEPA world model.

Pure-Python reimplementation of the game logic in ``snake.py``, decoupled from
pygame and NEAT. Exposes ``reset()`` / ``step(action)`` and a raw grid
observation (3 channels: head / body / apple), which is what the encoder learns
to represent.

Coordinates are integer grid cells: x in [0, GRID_W), y in [0, GRID_H).
The grid matches snake.py (800x400 px with 50 px cells -> 16 x 8 cells).
"""

from __future__ import annotations

import numpy as np

# --- Grid ---------------------------------------------------------------
GRID_W = 16  # columns (x)
GRID_H = 8   # rows (y)

# --- Actions ------------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
N_ACTIONS = 4

# (dx, dy) per action -- y grows downward, like the rendered board
MOVES = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Observation channels ----------------------------------------------
CH_HEAD, CH_BODY, CH_APPLE = 0, 1, 2
N_CHANNELS = 3

# --- Rewards ------------------------------------------------------------
R_FOOD = 1.0
R_DEATH = -1.0
R_STEP = 0.0
R_SHAPING = 0.1   # potential-based reward per cell closer/further to the apple


class SnakeEnv:
    """A minimal, deterministic-except-for-food Snake environment."""

    def __init__(self, max_steps: int = 500, shaping: float = R_SHAPING,
                 seed: int | None = None):
        self.max_steps = max_steps
        self.shaping = shaping
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial observation."""
        cx, cy = GRID_W // 2, GRID_H // 2
        self.snake: list[tuple[int, int]] = [(cx, cy)]  # head is index 0
        self.direction = RIGHT
        self.steps = 0
        self.done = False
        self._place_apple()
        return self._obs()

    def step(self, action: int):
        """Advance one step. Returns ``(obs, reward, done, info)``."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        # A 180-degree reversal is ignored (matches snake.py).
        if action != OPPOSITE[self.direction]:
            self.direction = action

        dx, dy = MOVES[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        prev_dist = self._apple_dist(hx, hy)

        # Collision with a wall or the body (conservative tail check, as in snake.py).
        hit_wall = not (0 <= nx < GRID_W and 0 <= ny < GRID_H)
        if hit_wall or (nx, ny) in self.snake:
            self.done = True
            return self._obs(), R_DEATH, True, self._info(ate=False, won=False)

        self.snake.insert(0, (nx, ny))
        ate = self.apple is not None and (nx, ny) == self.apple
        if ate:
            self._place_apple()          # snake grows: keep the tail
            reward = R_FOOD
        else:
            self.snake.pop()             # move: drop the tail
            # Dense, potential-based guidance toward the apple.
            reward = R_STEP + self.shaping * (prev_dist - self._apple_dist(nx, ny))

        self.steps += 1
        won = self.apple is None         # board fully filled
        self.done = won or self.steps >= self.max_steps
        return self._obs(), reward, self.done, self._info(ate=ate, won=won)

    # --- internals ------------------------------------------------------
    def _apple_dist(self, x: int, y: int) -> int:
        if self.apple is None:
            return 0
        return abs(x - self.apple[0]) + abs(y - self.apple[1])

    def _place_apple(self) -> None:
        free = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                if (x, y) not in self.snake]
        if not free:
            self.apple = None            # win state
        else:
            self.apple = free[int(self.rng.integers(len(free)))]

    def _obs(self) -> np.ndarray:
        obs = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[CH_HEAD, hy, hx] = 1.0
        for x, y in self.snake[1:]:
            obs[CH_BODY, y, x] = 1.0
        if self.apple is not None:
            ax, ay = self.apple
            obs[CH_APPLE, ay, ax] = 1.0
        return obs

    def _info(self, ate: bool, won: bool) -> dict:
        return {"score": self.score, "ate": ate, "won": won, "steps": self.steps}

    @property
    def score(self) -> int:
        """Number of apples eaten (snake length minus the initial head)."""
        return len(self.snake) - 1
