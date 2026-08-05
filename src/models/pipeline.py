"""End-to-end pipeline: oracle data -> train -> DAgger rounds -> evaluation.

    round 0 : safe-oracle dataset            -> train from scratch -> evaluate
    round i : + transitions played by the planner itself (merged with a fraction
              of the previous dataset) -> warm-start retrain -> evaluate

Everything is driven by ``src/config.py``, so every knob is settable through an
environment variable without touching the code:

    DAGGER_ROUNDS=4 MPC_HORIZON=6 python -m src.models.pipeline

Run:  python -m src.models.pipeline
"""

from __future__ import annotations

import json
import time
from datetime import date

import torch

from src.config import config
from src.data import make_dataset
from src.data.collect_with_planner import collect_with_planner, merge_datasets
from src.models import train as train_mod
from src.models.planner import load_world_model
from src.validation.metrics import evaluate_agent


def evaluate(model_path, device, episodes: int | None = None) -> dict:
    model = load_world_model(device, path=model_path)
    return evaluate_agent(model, device, episodes)


def main(rounds: int | None = None, epochs: int | None = None) -> dict:
    rounds = config.DAGGER_ROUNDS if rounds is None else rounds
    device = train_mod.resolve_device(config.DEVICE)
    dataset_path = config.DATA_PROCESSED / config.TRANSITIONS_FILE
    t0 = time.time()
    report = {"rounds": [], "config": {
        "env_max_steps": config.ENV_MAX_STEPS, "mpc_horizon": config.MPC_HORIZON,
        "mpc_space_coef": config.MPC_SPACE_COEF, "space_coef": config.SPACE_COEF,
        "embed_dim": config.EMBED_DIM, "rollout_k": config.ROLLOUT_K,
        "data_n_transitions": config.DATA_N_TRANSITIONS,
        "dagger_transitions": config.DAGGER_TRANSITIONS,
        "dagger_keep": config.DAGGER_KEEP, "epochs": epochs or config.EPOCHS,
    }}

    # --- Round 0: the safe oracle, the only source of long-snake states -----
    print("\n===== Round 0 : oracle dataset =====")
    data = make_dataset.build_dataset(config.DATA_N_TRANSITIONS, seed=0)
    make_dataset.save(data, dataset_path)
    print(f"[Data] {make_dataset.describe(data)}")

    out = train_mod.main(epochs=epochs, dataset=dataset_path, tag="round-0")
    agent = evaluate(out["model_path"], device)
    print(f"[Round 0] MPC mean={agent['mpc']['mean_score']} max={agent['mpc']['max_score']} "
          f"| greedy mean={agent['greedy']['mean_score']}")
    report["rounds"].append({"round": 0, "data": make_dataset.describe(data),
                             "best_val_head": out["best_val_head"], "agent": agent,
                             "model": str(out["model_path"])})

    best = {"round": 0, "mean": agent["mpc"]["mean_score"], "model": out["model_path"]}

    # --- DAgger rounds: train where the planner actually goes ---------------
    for r in range(1, rounds + 1):
        print(f"\n===== Round {r} : DAgger =====")
        fresh = collect_with_planner(out["model_path"], seed=100 * r)
        print(f"[DAgger] planner data: {make_dataset.describe(fresh)}")
        merged = merge_datasets(dataset_path, fresh)
        make_dataset.save(merged, dataset_path)

        out = train_mod.main(epochs=epochs, dataset=dataset_path,
                             init_from=out["model_path"], tag=f"round-{r}")
        agent = evaluate(out["model_path"], device)
        print(f"[Round {r}] MPC mean={agent['mpc']['mean_score']} "
              f"max={agent['mpc']['max_score']}")
        report["rounds"].append({"round": r, "data": make_dataset.describe(merged),
                                 "best_val_head": out["best_val_head"], "agent": agent,
                                 "model": str(out["model_path"])})

        if agent["mpc"]["mean_score"] > best["mean"]:
            best = {"round": r, "mean": agent["mpc"]["mean_score"],
                    "model": out["model_path"]}

    report["best"] = {"round": best["round"], "mean_score": best["mean"],
                      "model": str(best["model"])}
    report["duration_min"] = round((time.time() - t0) / 60, 2)

    path = config.OUTPUTS_RESULTS / f"pipeline_{date.today().isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n===== Pipeline done in {report['duration_min']} min =====")
    for row in report["rounds"]:
        print(f"  round {row['round']} : MPC mean {row['agent']['mpc']['mean_score']:<6} "
              f"max {row['agent']['mpc']['max_score']}")
    print(f"  best  : round {best['round']} (mean {best['mean']}) -> {best['model']}")
    print(f"  saved -> {path}")
    return report


if __name__ == "__main__":
    main()
