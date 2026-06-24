"""Validation metrics for the JEPA world model.

Four families:
  1. Latent health  -> collapse monitor (mean std + effective rank).
  2. Multi-step drift -> latent prediction error at each imagined horizon.
  3. Head quality   -> reward MSE (overall / on food / on death) and done
                       precision / recall / F1 (deaths are rare -> accuracy lies).
  4. Agent eval     -> let the MPC controller play, compare to random / greedy.

Run:  python -m src.validation.metrics
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from src.config import config
from src.envs.snake_env import SnakeEnv, N_ACTIONS
from src.data.make_dataset import behavior_action
from src.models.train import load_splits, resolve_device
from src.models.planner import LatentMPC, load_world_model, play_episode


# --------------------------------------------------------------------------- #
# 1. Latent health (collapse monitor)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latent_health(model, split, device, n: int = 4000) -> dict:
    idx = split.starts[:n]
    z = model.encode(split.obs[idx].to(device))
    zc = z - z.mean(dim=0)
    std = torch.sqrt(zc.var(dim=0) + 1e-4)
    # Effective rank = exp(entropy of normalized singular values).
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    return {"emb_std": float(std.mean()), "effective_rank": eff_rank,
            "embed_dim": int(z.shape[1])}


# --------------------------------------------------------------------------- #
# 2. Multi-step drift
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prediction_drift(model, split, device, n: int = 2000) -> list[float]:
    idx = split.starts[:n]
    s = model.encode(split.obs[idx].to(device))
    errs = []
    for k in range(split.k):
        a_k = split.action[idx + k].to(device)
        s = model.predict(s, a_k)
        target = model.encode_target(split.next_obs[idx + k].to(device))
        errs.append(round(float(F.mse_loss(s, target)), 5))
    return errs  # errs[k] = latent MSE at horizon k+1


# --------------------------------------------------------------------------- #
# 3. Head quality
# --------------------------------------------------------------------------- #
@torch.no_grad()
def head_quality(model, split, device, batch: int = 2048) -> dict:
    obs, act, rew, done = split.obs, split.action, split.reward, split.done
    r_pred, d_logit = [], []
    for b in range(0, len(act), batch):
        s = model.encode(obs[b:b + batch].to(device))
        a = act[b:b + batch].to(device)
        r_pred.append(model.reward(s, a).cpu())
        d_logit.append(model.done_logit(s, a).cpu())
    r_pred = torch.cat(r_pred)
    d_logit = torch.cat(d_logit)

    food, death = rew > 0.5, rew < -0.5
    reward = {
        "mse_all": round(float(F.mse_loss(r_pred, rew)), 4),
        "mse_food": round(float(F.mse_loss(r_pred[food], rew[food])), 4) if food.any() else None,
        "mse_death": round(float(F.mse_loss(r_pred[death], rew[death])), 4) if death.any() else None,
    }

    pred_pos, true_pos = d_logit > 0, done.bool()
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    done_metrics = {
        "deaths": int(true_pos.sum()), "n": int(len(done)),
        "recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3),
    }
    return {"reward": reward, "done": done_metrics}


# --------------------------------------------------------------------------- #
# 4. Agent evaluation (MPC vs baselines)
# --------------------------------------------------------------------------- #
def evaluate_agent(model, device, episodes: int = 30) -> dict:
    planner = LatentMPC(model, horizon=config.MPC_HORIZON,
                        gamma=config.MPC_GAMMA, device=device)
    rng = np.random.default_rng(0)
    policies = {
        "mpc": lambda e, o: planner.act(o),
        "random": lambda e, o: int(rng.integers(N_ACTIONS)),
        "greedy": lambda e, o: behavior_action(e, rng, epsilon=0.0),
    }
    out = {}
    for name, policy in policies.items():
        env = SnakeEnv(max_steps=500, seed=2026)
        eps = [play_episode(env, policy) for _ in range(episodes)]
        scores = [e["score"] for e in eps]
        out[name] = {"mean_score": round(float(np.mean(scores)), 2),
                     "max_score": int(np.max(scores)),
                     "mean_steps": round(float(np.mean([e["steps"] for e in eps])), 1)}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(episodes: int = 30) -> None:
    device = resolve_device(config.DEVICE)
    _, val_split, _ = load_splits(config, config.ROLLOUT_K)
    model = load_world_model(device)

    report = {
        "latent_health": latent_health(model, val_split, device),
        "drift_per_horizon": prediction_drift(model, val_split, device),
        "head_quality": head_quality(model, val_split, device),
        "agent": evaluate_agent(model, device, episodes),
    }

    print("\n===== JEPA world-model evaluation =====")
    h = report["latent_health"]
    print(f"Latent : emb_std={h['emb_std']:.3f} (collapse if ~0) | "
          f"effective_rank={h['effective_rank']:.1f}/{h['embed_dim']}")
    print(f"Drift  : latent MSE per horizon {report['drift_per_horizon']}")
    rq = report["head_quality"]
    print(f"Reward : MSE all={rq['reward']['mse_all']} | food={rq['reward']['mse_food']} | "
          f"death={rq['reward']['mse_death']}")
    print(f"Done   : recall={rq['done']['recall']} precision={rq['done']['precision']} "
          f"f1={rq['done']['f1']} ({rq['done']['deaths']} deaths / {rq['done']['n']})")
    print("Agent  :")
    for name, m in report["agent"].items():
        print(f"   {name:<7} mean={m['mean_score']:<5} max={m['max_score']:<3} "
              f"steps={m['mean_steps']}")

    out_dir = config.OUTPUTS_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"evaluation_{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
