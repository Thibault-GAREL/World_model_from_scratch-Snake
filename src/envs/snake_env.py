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

from src.config import config

# --- Grid ---------------------------------------------------------------
GRID_W = 16  # columns (x)
GRID_H = 8   # rows (y)
N_CELLS = GRID_W * GRID_H

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
R_SHAPING = config.R_SHAPING   # potential-based reward per cell closer/further


def reachable_cells(occupied: set[tuple[int, int]], start: tuple[int, int]) -> int:
    """Number of free cells reachable from ``start`` (flood fill, body = wall).

    This is the "am I about to trap myself?" signal. A greedy Snake player dies
    by walking into a pocket smaller than its own body, and a short-horizon
    planner cannot see that pocket either, so both need this measure.
    """
    x, y = start
    if not (0 <= x < GRID_W and 0 <= y < GRID_H) or start in occupied:
        return 0
    seen = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for dx, dy in MOVES.values():
            nxt = (cx + dx, cy + dy)
            if (0 <= nxt[0] < GRID_W and 0 <= nxt[1] < GRID_H
                    and nxt not in occupied and nxt not in seen):
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


class SnakeEnv:
    """A minimal, deterministic-except-for-food Snake environment."""

    def __init__(self, max_steps: int | None = None, shaping: float = R_SHAPING,
                 seed: int | None = None):
        self.max_steps = config.ENV_MAX_STEPS if max_steps is None else max_steps
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

    # --- free-space helpers (oracle safety + free-space head supervision) ---
    def legal_moves(self) -> list[int]:
        """Actions that are neither a 180 degree reversal nor an instant death."""
        out = []
        for a in range(N_ACTIONS):
            if a == OPPOSITE[self.direction]:
                continue
            dx, dy = MOVES[a]
            nx, ny = self.snake[0][0] + dx, self.snake[0][1] + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if (nx, ny) in self.snake:
                continue
            out.append(a)
        return out

    def free_fraction(self) -> float:
        """Fraction of the board reachable from the head (1.0 = wide open)."""
        if self.done:
            return 0.0
        body = set(self.snake[1:])
        return reachable_cells(body, self.snake[0]) / N_CELLS

    def free_fraction_after(self, action: int) -> float:
        """``free_fraction`` of the state that would follow ``action`` (0.0 if it kills).

        Used by the oracle to refuse moves that lead into a pocket, without
        mutating the environment.
        """
        direction = self.direction if action == OPPOSITE[self.direction] else action
        dx, dy = MOVES[direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < GRID_W and 0 <= ny < GRID_H) or (nx, ny) in self.snake:
            return 0.0
        ate = self.apple is not None and (nx, ny) == self.apple
        body = set(self.snake if ate else self.snake[:-1])   # tail frees up unless we eat
        return reachable_cells(body, (nx, ny)) / N_CELLS
