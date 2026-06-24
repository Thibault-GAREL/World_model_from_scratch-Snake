"""Build the world-model dataset: (o_t, a_t, r_t, done, o_{t+1}) transitions.

Transitions are collected with a *mixed* behaviour policy (epsilon-random +
greedy-toward-apple with a one-step safety check). Pure random play dies too
fast and almost never eats, which would starve the reward head of positive
examples; the greedy bias yields longer, more diverse episodes.

Transitions are stored in episode order with ``done`` flags, so the training
loop can later sample K-step windows for multi-step training without crossing
an episode boundary.

Run:  python -m src.data.make_dataset
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.config import config
from src.envs.snake_env import SnakeEnv, GRID_W, GRID_H, MOVES, OPPOSITE, N_ACTIONS


def behavior_action(env: SnakeEnv, rng: np.random.Generator, epsilon: float) -> int:
    """Epsilon-random, else greedy toward the apple avoiding immediate death."""
    if env.apple is None or rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    hx, hy = env.snake[0]
    ax, ay = env.apple
    best_action, best_dist = None, None
    for a in range(N_ACTIONS):
        if a == OPPOSITE[env.direction]:
            continue
        dx, dy = MOVES[a]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
            continue          # would hit a wall
        if (nx, ny) in env.snake:
            continue          # would hit the body
        dist = abs(nx - ax) + abs(ny - ay)
        if best_dist is None or dist < best_dist:
            best_dist, best_action = dist, a

    if best_action is None:   # no safe move -> any non-reversal action
        return int(rng.integers(N_ACTIONS))
    return best_action


def main(n_transitions: int = 50_000, epsilon: float = 0.25,
         max_steps: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    env = SnakeEnv(max_steps=max_steps, seed=seed)

    obs_buf = np.empty((n_transitions, 3, GRID_H, GRID_W), dtype=np.float32)
    next_buf = np.empty((n_transitions, 3, GRID_H, GRID_W), dtype=np.float32)
    act_buf = np.empty(n_transitions, dtype=np.int64)
    rew_buf = np.empty(n_transitions, dtype=np.float32)
    done_buf = np.empty(n_transitions, dtype=np.bool_)

    obs = env.reset()
    apples = episodes = 0
    for i in tqdm(range(n_transitions), desc="Collecting transitions"):
        action = behavior_action(env, rng, epsilon)
        next_obs, reward, done, info = env.step(action)

        obs_buf[i] = obs
        act_buf[i] = action
        rew_buf[i] = reward
        done_buf[i] = done
        next_buf[i] = next_obs

        apples += int(info["ate"])
        if done:
            episodes += 1
            obs = env.reset()
        else:
            obs = next_obs

    out_path = config.DATA_PROCESSED / "snake_transitions.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_buf, action=act_buf, reward=rew_buf, done=done_buf, next_obs=next_buf,
    )

    print(f"Saved {n_transitions} transitions over {episodes} episodes "
          f"({apples} apples eaten) -> {out_path}")


if __name__ == "__main__":
    main()
