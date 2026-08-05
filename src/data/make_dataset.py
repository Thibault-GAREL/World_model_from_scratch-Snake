"""Build the world-model dataset: (o_t, a_t, r_t, done, free, o_{t+1}) transitions.

The behaviour policy decides what the world model can ever learn. A model never
shown a 30-apple snake cannot predict its dynamics, and the planner then cannot
plan there: the agent's ceiling is its dataset's ceiling.

So episodes are of two kinds, mixed:

* **clean episodes** (``DATA_CLEAN_FRACTION``) played by the safe oracle, which
  walks toward the apple but refuses any move that would trap it in a pocket
  smaller than its own body (flood fill). These are what produce *long* snakes.
* **noisy episodes** with epsilon-random actions, which produce the varied
  deaths the done head needs (deaths are what the planner must learn to avoid).

``free`` stores the fraction of the board still reachable after the action, the
supervision target of the free-space head.

Transitions are stored in episode order with ``done`` flags, so the training
loop can later sample K-step windows without crossing an episode boundary.

Run:  python -m src.data.make_dataset
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from tqdm import tqdm

from src.config import config
from src.envs.snake_env import (
    SnakeEnv, GRID_W, GRID_H, MOVES, OPPOSITE, N_ACTIONS, N_CELLS,
)


# --------------------------------------------------------------------------- #
# Behaviour policies
# --------------------------------------------------------------------------- #
def behavior_action(env: SnakeEnv, rng: np.random.Generator, epsilon: float,
                    safe: bool | None = None, safe_explore: float = 0.0) -> int:
    """Greedy toward the apple with a flood-fill safety check, plus two noises.

    The safety check is the whole difference between a snake that plateaus
    around 15 apples and one that reaches 40+: a purely greedy player walks into
    a pocket it cannot get out of as soon as its body gets long.

    The two noises play different roles and both are needed:

    * ``epsilon`` picks a fully random action, which sometimes kills. Those are
      the only positive examples the done head ever gets.
    * ``safe_explore`` picks a random *legal* action. A pure oracle plays one
      action per state, so the model never sees what the other three do, yet
      the planner scores all four at every step and reads pure extrapolation
      there. This noise covers the (state, action) space without dying.
    """
    if safe is None:
        safe = config.DATA_ORACLE_SAFE

    if env.apple is None or rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))       # may be suicidal, on purpose

    moves = env.legal_moves()
    if not moves:
        return int(rng.integers(N_ACTIONS))       # doomed whatever we do

    if safe_explore and rng.random() < safe_explore:
        return int(moves[rng.integers(len(moves))])

    ax, ay = env.apple
    hx, hy = env.snake[0]
    need = len(env.snake) + 1                     # cells needed to avoid self-trapping

    scored = []
    for a in moves:
        dx, dy = MOVES[a]
        nx, ny = hx + dx, hy + dy
        dist = abs(nx - ax) + abs(ny - ay)
        space = env.free_fraction_after(a) * N_CELLS if safe else N_CELLS
        scored.append((a, dist, space))

    roomy = [t for t in scored if t[2] >= need]
    if roomy:                                     # enough room: head for the apple
        return min(roomy, key=lambda t: (t[1], -t[2]))[0]
    return max(scored, key=lambda t: t[2])[0]     # cornered: buy space instead


def fixed_policy(epsilon: float, safe_explore: float | None = None):
    """Behaviour policy at a constant epsilon and safe-exploration rate."""
    if safe_explore is None:
        safe_explore = config.DATA_SAFE_EXPLORE

    def act(env: SnakeEnv, obs: np.ndarray, rng: np.random.Generator) -> int:
        return behavior_action(env, rng, epsilon, safe_explore=safe_explore)
    return act


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def collect(policy: Callable, n_transitions: int, max_steps: int | None = None,
            seed: int = 0, desc: str = "Collecting transitions") -> dict:
    """Roll ``policy`` in the env and return the transition arrays.

    ``policy(env, obs, rng) -> action``. An optional ``policy.on_reset(rng)``
    hook is called at every episode start (used to draw a per-episode epsilon).
    """
    rng = np.random.default_rng(seed)
    env = SnakeEnv(max_steps=max_steps, seed=seed)

    obs_buf = np.empty((n_transitions, 3, GRID_H, GRID_W), dtype=np.float32)
    next_buf = np.empty((n_transitions, 3, GRID_H, GRID_W), dtype=np.float32)
    act_buf = np.empty(n_transitions, dtype=np.int64)
    rew_buf = np.empty(n_transitions, dtype=np.float32)
    done_buf = np.empty(n_transitions, dtype=np.bool_)
    free_buf = np.empty(n_transitions, dtype=np.float32)

    on_reset = getattr(policy, "on_reset", None)
    obs = env.reset()
    if on_reset:
        on_reset(rng)

    apples = episodes = 0
    scores: list[int] = []
    for i in tqdm(range(n_transitions), desc=desc):
        action = policy(env, obs, rng)
        next_obs, reward, done, info = env.step(action)

        obs_buf[i] = obs
        act_buf[i] = action
        rew_buf[i] = reward
        done_buf[i] = done
        next_buf[i] = next_obs
        free_buf[i] = env.free_fraction()          # 0.0 when the episode just ended

        apples += int(info["ate"])
        if done:
            episodes += 1
            scores.append(info["score"])
            obs = env.reset()
            if on_reset:
                on_reset(rng)
        else:
            obs = next_obs

    return {
        "obs": obs_buf, "action": act_buf, "reward": rew_buf, "done": done_buf,
        "free": free_buf, "next_obs": next_buf,
        "_stats": {"episodes": episodes, "apples": apples,
                   "mean_score": float(np.mean(scores)) if scores else 0.0,
                   "max_score": int(np.max(scores)) if scores else 0},
    }


def save(data: dict, path=None) -> None:
    path = path or config.DATA_PROCESSED / config.TRANSITIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=data["obs"], action=data["action"], reward=data["reward"],
        done=data["done"], free=data["free"], next_obs=data["next_obs"],
    )
    return path


def describe(data: dict) -> str:
    """One-line coverage summary: what snake lengths the model will ever see."""
    score = data["obs"][:, 1].sum(axis=(1, 2))     # body cells = apples eaten
    st = data["_stats"]
    return (f"{len(score)} transitions | {st['episodes']} episodes "
            f"| episode score mean {st['mean_score']:.1f} max {st['max_score']} "
            f"| states p50={np.percentile(score, 50):.0f} "
            f"p95={np.percentile(score, 95):.0f} max={score.max():.0f} "
            f"| deaths {100 * data['done'].mean():.2f}%")


def build_dataset(n_transitions: int | None = None, seed: int = 0) -> dict:
    """Mix clean-oracle and noisy transitions, with quotas counted in *transitions*.

    Quotas must be per transition, not per episode: a clean episode lasts ~350
    steps and a noisy one ~20, so splitting the *episodes* 60/40 yields 98%
    clean transitions and a dataset with 0.3% deaths. The done head then never
    learns to die, and the planner walks straight into the wall.
    """
    n = n_transitions if n_transitions is not None else config.DATA_N_TRANSITIONS
    n_clean = int(n * config.DATA_CLEAN_FRACTION)
    n_noisy = n - n_clean

    parts = []
    if n_clean:
        parts.append(collect(fixed_policy(0.0), n_clean, seed=seed,
                             desc="Oracle transitions (long snakes)"))
    if n_noisy:
        parts.append(collect(fixed_policy(config.DATA_EPSILON), n_noisy, seed=seed + 1,
                             desc="Noisy transitions (deaths)"))

    merged = {k: np.concatenate([p[k] for p in parts])
              for k in ("obs", "action", "reward", "done", "free", "next_obs")}
    cut = np.cumsum([len(p["action"]) for p in parts]) - 1
    merged["done"][cut] = True                    # no K-step window across the seam
    merged["_stats"] = {
        "episodes": sum(p["_stats"]["episodes"] for p in parts),
        "apples": sum(p["_stats"]["apples"] for p in parts),
        "mean_score": float(np.mean([p["_stats"]["mean_score"] for p in parts])),
        "max_score": max(p["_stats"]["max_score"] for p in parts),
    }
    return merged


def main(n_transitions: int | None = None, seed: int = 0) -> None:
    data = build_dataset(n_transitions, seed)
    path = save(data)
    print(f"[Dataset] {describe(data)}")
    print(f"[Dataset] saved -> {path}")


if __name__ == "__main__":
    main()
