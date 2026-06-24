"""Unit tests for the headless Snake environment."""

import numpy as np

from src.envs.snake_env import (
    SnakeEnv,
    GRID_W,
    GRID_H,
    N_CHANNELS,
    RIGHT,
    LEFT,
)


def test_reset_obs_shape():
    env = SnakeEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (N_CHANNELS, GRID_H, GRID_W)
    assert obs.dtype == np.float32
    assert obs[0].sum() == 1.0   # exactly one head
    assert obs[2].sum() == 1.0   # exactly one apple


def test_death_on_wall():
    env = SnakeEnv(seed=0)
    env.reset()  # starts at the center, moving RIGHT
    done, r = False, 0.0
    for _ in range(GRID_W + 2):
        _, r, done, _ = env.step(RIGHT)
        if done:
            break
    assert done
    assert r == -1.0


def test_reverse_is_ignored():
    env = SnakeEnv(seed=1)
    env.reset()  # direction RIGHT
    # Asking to go LEFT (a 180 reversal) must be ignored -> keep going RIGHT.
    _, _, done, _ = env.step(LEFT)
    assert not done  # would have instantly self-collided if it had reversed


def test_random_rollout_is_stable():
    rng = np.random.default_rng(0)
    env = SnakeEnv(seed=0)
    env.reset()
    for _ in range(2000):
        obs, r, done, info = env.step(int(rng.integers(4)))
        assert obs.shape == (N_CHANNELS, GRID_H, GRID_W)
        assert -1.0 <= r <= 1.0          # death, food, or small shaping reward
        if done:
            env.reset()
