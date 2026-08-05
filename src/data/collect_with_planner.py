"""DAgger-style data collection: let the planner visit its own states.

The oracle dataset teaches the model the dynamics of states *the oracle* visits.
The planner visits different ones (it is worse, and wrong in its own way), and
in those states the model was never trained, so it predicts badly, so it plans
badly. That is textbook distribution shift, the exact problem DAgger solves in
the Decision Tree version of this series.

Here the fix is the same idea in latent space: play with the current planner,
record what it actually encounters, fold that into the dataset, retrain. The
labels need no oracle, the environment itself provides reward / done / free.

Collection is CPU-parallel: one worker per core, each with its own env, its own
seed and its own copy of the (tiny) model.

Run:  python -m src.data.collect_with_planner
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.config import config
from src.data.make_dataset import collect, describe, save
from src.envs.snake_env import N_ACTIONS

ARRAYS = ("obs", "action", "reward", "done", "free", "next_obs")


def planner_policy(planner, epsilon: float):
    """Play with the planner, with a little noise to keep the states diverse."""
    def act(env, obs, rng):
        if rng.random() < epsilon:
            return int(rng.integers(N_ACTIONS))
        return planner.act(obs)
    return act


def _worker(args) -> dict:
    """One collection shard, in its own process (torch limited to one thread)."""
    model_path, n, epsilon, seed, horizon, space_coef = args
    torch.set_num_threads(1)                   # N processes x 1 thread beats the reverse
    from src.models.planner import LatentMPC, load_world_model

    device = torch.device("cpu")
    model = load_world_model(device, path=model_path)
    planner = LatentMPC(model, horizon=horizon, gamma=config.MPC_GAMMA,
                        device=device, space_coef=space_coef)
    data = collect(planner_policy(planner, epsilon), n, seed=seed,
                   desc=f"Planner shard (seed {seed})")
    return data


def collect_with_planner(model_path, n_transitions: int | None = None,
                         n_workers: int | None = None, epsilon: float | None = None,
                         seed: int = 100) -> dict:
    """Collect ``n_transitions`` played by the planner, sharded over processes."""
    n_transitions = config.DAGGER_TRANSITIONS if n_transitions is None else n_transitions
    epsilon = config.DAGGER_EPSILON if epsilon is None else epsilon
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers, n_transitions // 2000 or 1))

    base, rest = divmod(n_transitions, n_workers)
    shards = [base + (1 if i < rest else 0) for i in range(n_workers)]
    args = [(str(model_path), shard, epsilon, seed + i,
             config.MPC_HORIZON, config.MPC_SPACE_COEF)
            for i, shard in enumerate(shards)]

    print(f"[DAgger] collecting {n_transitions} planner transitions "
          f"on {n_workers} workers (epsilon={epsilon})")
    if n_workers == 1:
        parts = [_worker(args[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_worker, args))

    merged = {k: np.concatenate([p[k] for p in parts]) for k in ARRAYS}
    # Each shard ends mid-episode: cut there so no K-step window spans two shards.
    cut = np.cumsum([len(p["action"]) for p in parts]) - 1
    merged["done"][cut] = True

    stats = [p["_stats"] for p in parts]
    merged["_stats"] = {
        "episodes": sum(s["episodes"] for s in stats),
        "apples": sum(s["apples"] for s in stats),
        "mean_score": float(np.mean([s["mean_score"] for s in stats])),
        "max_score": max(s["max_score"] for s in stats),
    }
    return merged


def merge_datasets(old_path, new_data: dict, keep: float | None = None) -> dict:
    """Keep a fraction of the previous dataset and append the fresh one.

    Dropping the old data entirely makes the model forget the oracle's long
    snakes, keeping all of it drowns the new states. ``DAGGER_KEEP`` is that
    trade-off.
    """
    keep = config.DAGGER_KEEP if keep is None else keep
    old_path = Path(old_path)
    if not old_path.exists() or keep <= 0:
        return new_data

    old = np.load(old_path)
    n_keep = int(len(old["action"]) * keep)
    if n_keep <= 0:
        return new_data

    parts = {}
    for k in ARRAYS:
        if k not in old.files:                 # dataset predating the free-space head
            head = np.zeros(n_keep, dtype=np.float32)
        else:
            head = old[k][:n_keep]
        parts[k] = np.concatenate([head, new_data[k]])
    parts["done"] = parts["done"].copy()
    parts["done"][n_keep - 1] = True           # close the truncated episode
    parts["_stats"] = new_data["_stats"]
    return parts


def main(model_path=None) -> None:
    from src.models.planner import find_latest_model

    model_path = model_path or find_latest_model()
    data = collect_with_planner(model_path)
    print(f"[DAgger] planner data: {describe(data)}")
    out = config.DATA_PROCESSED / "planner_transitions.npz"
    save(data, out)
    print(f"[DAgger] saved -> {out}")


if __name__ == "__main__":
    main()
