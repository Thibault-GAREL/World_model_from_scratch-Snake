"""Boucle d'entraînement du JEPA world model + logging MLflow.

Loss = PRED_COEF · ‖ŝ − sg(target)‖²            (latent prediction, multi-step)
     + STD_COEF · variance hinge + COV_COEF · covariance   (VICReg, anti-collapse)
     + REWARD_COEF · MSE(reward) + DONE_COEF · BCE(done, pos_weight)

The target encoder is an EMA copy of the encoder (stop-gradient). A collapse
monitor (mean embedding std) is logged every epoch - if it falls toward 0 the
encoder is collapsing. The done head uses a class-balanced ``pos_weight`` because
deaths are rare (~2% of transitions), so plain accuracy is meaningless; we track
recall/precision instead.

Run:  python -m src.models.train
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import config
from src.models.model import WorldModel


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def next_run_dir(base: Path, model_name: str) -> Path:
    """outputs/.../{model}_run-NN_date-YYYY-MM-DD  (NN auto-incremented)."""
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.name.startswith(model_name + "_run-")]
    runs = [int(n.split("_run-")[1][:2]) for n in existing] if existing else []
    nn = (max(runs) + 1) if runs else 1
    return base / f"{model_name}_run-{nn:02d}_date-{date.today().isoformat()}"


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(z: torch.Tensor):
    """Variance hinge + covariance penalty on a batch of embeddings (B, D)."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std))
    cov = (z.T @ z) / (z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow(2).sum() / z.shape[1]
    return std_loss, cov_loss, std.mean().detach()


# --------------------------------------------------------------------------- #
# Data: episode-aware multi-step windows
# --------------------------------------------------------------------------- #
class Split:
    """Holds a contiguous slice of transitions and its valid K-step start indices."""

    def __init__(self, obs, action, reward, done, next_obs, k: int):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        # A window [i, i+K) is valid iff none of its first K-1 steps ends an episode.
        n = len(action)
        done_np = done.numpy()
        if k == 1:
            starts = np.arange(n)
        else:
            valid = [not done_np[i:i + k - 1].any() for i in range(n - k + 1)]
            starts = np.nonzero(valid)[0]
        self.starts = starts
        self.k = k

    def batches(self, batch_size: int, shuffle: bool, device):
        idx = self.starts.copy()
        if shuffle:
            np.random.shuffle(idx)
        for b in range(0, len(idx), batch_size):
            s = idx[b:b + batch_size]
            obs0 = self.obs[s].to(device)
            actions = torch.stack([self.action[s + k] for k in range(self.k)], dim=1).to(device)
            rewards = torch.stack([self.reward[s + k] for k in range(self.k)], dim=1).to(device)
            dones = torch.stack([self.done[s + k].float() for k in range(self.k)], dim=1).to(device)
            next_seq = torch.stack([self.next_obs[s + k] for k in range(self.k)], dim=1).to(device)
            yield obs0, actions, rewards, dones, next_seq


def load_splits(cfg, k: int):
    path = config.DATA_PROCESSED / cfg.TRANSITIONS_FILE
    data = np.load(path)
    obs = torch.from_numpy(data["obs"])
    action = torch.from_numpy(data["action"])
    reward = torch.from_numpy(data["reward"])
    done = torch.from_numpy(data["done"])
    next_obs = torch.from_numpy(data["next_obs"])

    n = len(action)
    n_val = int(n * cfg.VAL_FRACTION)
    n_train = n - n_val
    tr = Split(obs[:n_train], action[:n_train], reward[:n_train], done[:n_train],
               next_obs[:n_train], k)
    va = Split(obs[n_train:], action[n_train:], reward[n_train:], done[n_train:],
               next_obs[n_train:], k)
    return tr, va, n


def done_pos_weight(split, cap: float, device) -> torch.Tensor:
    """(#negatives / #positives) for the done BCE, capped to avoid extremes."""
    d = split.done.float()
    n_pos = float(d.sum())
    n_neg = float(len(d) - n_pos)
    w = min(n_neg / max(n_pos, 1.0), cap)
    return torch.tensor(w, device=device)


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def run_epoch(model, split, cfg, device, pos_weight, optimizer=None):
    train = optimizer is not None
    model.train(train)
    agg = {"loss": 0.0, "head": 0.0, "pred": 0.0, "reward": 0.0, "done": 0.0, "emb_std": 0.0}
    tp = fp = fn = 0
    n_batches = 0

    for obs0, actions, rewards, dones, next_seq in split.batches(
            cfg.BATCH_SIZE, shuffle=train, device=device):
        with torch.set_grad_enabled(train):
            s_pred = model.encode(obs0)
            std_loss, cov_loss, emb_std = vicreg_terms(s_pred)
            s_true = s_pred                                       # true latent at step k

            pred_l = rew_l = done_l = 0.0
            for k in range(cfg.ROLLOUT_K):
                a_k = actions[:, k]
                # --- Heads supervised on the TRUE encoded latent (clean signal,
                #     including death states). Weighted MSE: rare food/death events
                #     count more than the many 0-reward steps.
                r_pred, r_tgt = model.reward(s_true, a_k), rewards[:, k]
                w = 1.0 + cfg.REWARD_EVENT_WEIGHT * (r_tgt.abs() > 0.5).float()
                rew_l = rew_l + (w * (r_pred - r_tgt) ** 2).mean()
                logit = model.done_logit(s_true, a_k)
                done_l = done_l + F.binary_cross_entropy_with_logits(
                    logit, dones[:, k], pos_weight=pos_weight)

                pred_pos = logit > 0
                true_pos = dones[:, k] > 0.5
                tp += int((pred_pos & true_pos).sum())
                fp += int((pred_pos & ~true_pos).sum())
                fn += int((~pred_pos & true_pos).sum())

                # --- Dynamics: predicted latent vs EMA target (stop-grad).
                s_pred = model.predict(s_pred, a_k)               # ŝ_{k+1}
                target = model.encode_target(next_seq[:, k])
                pred_l = pred_l + F.mse_loss(s_pred, target)

                # Advance the TRUE latent for the next step's head supervision.
                if k < cfg.ROLLOUT_K - 1:
                    s_true = model.encode(next_seq[:, k])

            pred_l = pred_l / cfg.ROLLOUT_K
            rew_l = rew_l / cfg.ROLLOUT_K
            done_l = done_l / cfg.ROLLOUT_K
            # Head loss (fixed-scale targets) drives model selection. The latent
            # prediction MSE is NOT scale-invariant (it grows as VICReg expands the
            # latent), so it must not drive selection - only reward/done do.
            head = cfg.REWARD_COEF * rew_l + cfg.DONE_COEF * done_l
            loss = (cfg.PRED_COEF * pred_l + cfg.STD_COEF * std_loss
                    + cfg.COV_COEF * cov_loss + head)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            model.ema_update(cfg.EMA_DECAY)

        agg["loss"] += loss.item()
        agg["head"] += head.item()
        agg["pred"] += pred_l.item()
        agg["reward"] += rew_l.item()
        agg["done"] += done_l.item()
        agg["emb_std"] += emb_std.item()
        n_batches += 1

    out = {key: val / max(n_batches, 1) for key, val in agg.items()}
    out["done_recall"] = tp / max(tp + fn, 1)
    out["done_precision"] = tp / max(tp + fp, 1)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(epochs: int | None = None) -> None:
    cfg = config
    n_epochs = epochs if epochs is not None else cfg.EPOCHS
    set_seed(cfg.RANDOM_STATE)
    device = resolve_device(cfg.DEVICE)

    train_split, val_split, n_total = load_splits(cfg, cfg.ROLLOUT_K)
    pos_weight = done_pos_weight(train_split, cfg.DONE_POS_WEIGHT_CAP, device)
    model = WorldModel(dim=cfg.EMBED_DIM, hidden=cfg.HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print(f"[Data] {n_total} transitions | train windows {len(train_split.starts)} "
          f"| val windows {len(val_split.starts)} | device {device} "
          f"| done pos_weight {float(pos_weight):.1f}")

    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({
            "embed_dim": cfg.EMBED_DIM, "hidden_dim": cfg.HIDDEN_DIM,
            "lr": cfg.LEARNING_RATE, "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE, "epochs": n_epochs,
            "rollout_k": cfg.ROLLOUT_K, "ema_decay": cfg.EMA_DECAY,
            "pred_coef": cfg.PRED_COEF, "std_coef": cfg.STD_COEF,
            "cov_coef": cfg.COV_COEF, "reward_coef": cfg.REWARD_COEF,
            "done_coef": cfg.DONE_COEF, "done_pos_weight": float(pos_weight),
            "seed": cfg.RANDOM_STATE, "n_transitions": n_total,
        })

        model_dir = next_run_dir(config.OUTPUTS_MODELS, cfg.MODEL_NAME)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_head = float("inf")   # select on the head loss (scale-stable), not the total
        bad_epochs = 0
        history = []

        for epoch in tqdm(range(n_epochs), desc="Training"):
            tr = run_epoch(model, train_split, cfg, device, pos_weight, optimizer)
            va = run_epoch(model, val_split, cfg, device, pos_weight, optimizer=None)

            mlflow.log_metrics({
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "train_head": tr["head"], "val_head": va["head"],
                "val_pred": va["pred"], "val_reward_mse": va["reward"],
                "val_done_recall": va["done_recall"], "val_done_precision": va["done_precision"],
                "emb_std": tr["emb_std"],            # collapse monitor (-> ~1, not 0)
            }, step=epoch)
            history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                            **{f"val_{k}": v for k, v in va.items()}})

            if va["head"] < best_head - 1e-4:
                best_head = va["head"]
                bad_epochs = 0
                torch.save(model.state_dict(), model_dir / "best_model.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.PATIENCE:
                    print(f"[EarlyStop] no val head improvement for {cfg.PATIENCE} epochs "
                          f"(epoch {epoch}).")
                    break

        results_dir = next_run_dir(config.OUTPUTS_RESULTS, cfg.MODEL_NAME)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump({"best_val_head": best_head, "history": history}, f, indent=2)

        mlflow.log_metric("best_val_head", best_head)
        last = history[-1]
        print(f"[Done] best val_head={best_head:.4f} | model -> {model_dir}")
        print(f"[Monitor] emb_std={last['train_emb_std']:.3f} (collapse if ~0) | "
              f"val done recall={last['val_done_recall']:.3f} "
              f"precision={last['val_done_precision']:.3f} | "
              f"val reward_mse={last['val_reward']:.4f}")


if __name__ == "__main__":
    main()
