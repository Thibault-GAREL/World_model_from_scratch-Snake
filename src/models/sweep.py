"""Parallel hyperparameter sweep: train + evaluate one config per process.

The model trains in about a minute, so the useful unit of work is not a longer
run, it is *more* runs. Each job is a subprocess driven entirely by environment
variables (the Pydantic config reads them), so no code is edited to sweep.

    python -m src.models.sweep                  # default grid
    python -m src.models.sweep --workers 12
    python -m src.models.sweep --grid my_grid.json

Memory note: each job holds its own copy of the dataset (roughly 0.7 GB for
200k transitions), so the worker count is capped well below the core count.
Data collection (``collect_with_planner``) is the part that scales to all cores.

Run:  python -m src.models.sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from src.config import config

# Each entry is an env-var name -> list of values to try.
#
# The measured bottleneck is the reward head: it ranks the four actions right
# only 36% of the time (25% is chance), so the planner survives but wanders.
# This grid therefore attacks reward learnability first (loss weights, shaping
# scale, latent capacity) rather than planner settings, which are useless until
# the model can tell a good action from a bad one.
DEFAULT_GRID = {
    "REWARD_COEF": [10.0, 50.0],       # weight of the reward head in the loss
    "PRED_COEF": [5.0, 25.0],          # latent prediction, currently dominates
    "EMBED_DIM": [128, 256],           # capacity to encode head/apple geometry
    "R_SHAPING": [0.2, 0.4],           # scale of the dense guidance signal
    "REWARD_EVENT_WEIGHT": [1.0, 3.0],  # extra weight on food/death vs shaping
}

# Planner-side grid, worth running once a model actually ranks actions well.
PLANNER_GRID = {
    "MPC_HORIZON": [4, 5, 6],
    "MPC_SPACE_COEF": [0.0, 0.3, 1.0],
    "ROLLOUT_K": [5, 8],
}

SWEEP_DIR = config.OUTPUTS_RESULTS / "sweep"


def job_name(combo: dict) -> str:
    short = {"MPC_HORIZON": "h", "MPC_SPACE_COEF": "sp", "ROLLOUT_K": "k",
             "EMBED_DIM": "d", "SPACE_COEF": "sc", "LEARNING_RATE": "lr",
             "EMA_DECAY": "ema", "PRED_COEF": "pc"}
    parts = [f"{short.get(k, k.lower())}{v}" for k, v in combo.items()]
    return "-".join(parts).replace(".", "p")       # 10.0 -> "10p0", never "100"


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


# --------------------------------------------------------------------------- #
# One job (runs in its own process)
# --------------------------------------------------------------------------- #
def run_job() -> None:
    """Train on the existing dataset, evaluate the planner, write a JSON row."""
    import torch
    from src.models import train as train_mod
    from src.models.planner import load_world_model
    from src.validation.metrics import action_ranking, evaluate_agent, latent_health

    torch.set_num_threads(int(os.environ.get("JOB_THREADS", "2")))
    name = os.environ["JOB_NAME"]
    t0 = time.time()

    out = train_mod.main(tag=name)
    device = train_mod.resolve_device(config.DEVICE)
    model = load_world_model(device, path=out["model_path"])
    _, val_split, _ = train_mod.load_splits(config, config.ROLLOUT_K)

    row = {
        "name": name,
        "params": {k: os.environ[k] for k in os.environ.get("JOB_KEYS", "").split(",") if k},
        "best_val_head": out["best_val_head"],
        "latent": latent_health(model, val_split, device),
        "ranking": action_ranking(model, device),
        "agent": evaluate_agent(model, device),
        "minutes": round((time.time() - t0) / 60, 2),
        "model": str(out["model_path"]),
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DIR / f"{name}.json", "w") as f:
        json.dump(row, f, indent=2)
    print(f"[{name}] MPC mean={row['agent']['mpc']['mean_score']} "
          f"max={row['agent']['mpc']['max_score']} ({row['minutes']} min)")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def launch(combo: dict, threads: int) -> tuple[str, int]:
    name = job_name(combo)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in combo.items()})
    env["JOB_NAME"] = name
    env["JOB_KEYS"] = ",".join(combo)
    env["JOB_THREADS"] = str(threads)
    env["MODEL_NAME"] = f"sweep-{name}"        # keeps run dirs from colliding
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"

    log_dir = config.OUTPUTS_LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"sweep-{name}.log", "w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "src.models.sweep", "--job"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"  {name:<28} {status}")
    return name, proc.returncode


def main(workers: int | None = None, grid_path: str | None = None) -> None:
    grid = json.loads(Path(grid_path).read_text()) if grid_path else DEFAULT_GRID
    combos = expand(grid)

    cores = os.cpu_count() or 4
    workers = workers or max(1, min(cores // 2, 12))
    threads = max(1, cores // max(workers, 1))

    dataset = config.DATA_PROCESSED / config.TRANSITIONS_FILE
    if not dataset.exists():
        print(f"[Sweep] no dataset at {dataset}, building it first")
        from src.data import make_dataset
        make_dataset.main()

    print(f"[Sweep] {len(combos)} configs | {workers} workers x {threads} threads "
          f"| {cores} cores available")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda c: launch(c, threads), combos))

    rows = []
    for path in sorted(SWEEP_DIR.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda r: r["agent"]["mpc"]["mean_score"], reverse=True)

    summary = {"date": date.today().isoformat(), "grid": grid,
               "minutes": round((time.time() - t0) / 60, 2), "rows": rows}
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== Sweep done in {summary['minutes']} min =====")
    print(f"{'config':<34}{'MPC mean':>10}{'max':>6}{'rank%':>8}{'safe%':>8}")
    for row in rows[:15]:
        mpc, rk = row["agent"]["mpc"], row.get("ranking", {})
        print(f"{row['name']:<34}{mpc['mean_score']:>10}{mpc['max_score']:>6}"
              f"{100 * rk.get('best_action_agreement', 0):>7.0f}%"
              f"{100 * rk.get('safest_action_is_safe', 0):>7.0f}%")
    print(f"\nSaved -> {SWEEP_DIR / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="store_true", help="internal: run one config")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--grid", type=str, default=None, help="JSON file overriding the grid")
    args = parser.parse_args()

    if args.job:
        run_job()
    else:
        main(workers=args.workers, grid_path=args.grid)
