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

    def __init__(self, obs, action, reward, done, next_obs, k: int, free=None):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.free = torch.zeros_like(reward) if free is None else free
        # A window [i, i+K) is valid iff none of its first K-1 steps ends an episode.
        n = len(action)
        done_np = done.cpu().numpy()
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
        home = self.obs.device            # where the dataset lives (RAM or VRAM)
        for b in range(0, len(idx), batch_size):
            s = torch.as_tensor(idx[b:b + batch_size], device=home)
            obs0 = self.obs[s].to(device)
            actions = torch.stack([self.action[s + k] for k in range(self.k)], dim=1).to(device)
            rewards = torch.stack([self.reward[s + k] for k in range(self.k)], dim=1).to(device)
            dones = torch.stack([self.done[s + k].float() for k in range(self.k)], dim=1).to(device)
            frees = torch.stack([self.free[s + k] for k in range(self.k)], dim=1).to(device)
            next_seq = torch.stack([self.next_obs[s + k] for k in range(self.k)], dim=1).to(device)
            yield obs0, actions, rewards, dones, frees, next_seq


def load_splits(cfg, k: int, path=None, device=None):
    path = path or config.DATA_PROCESSED / cfg.TRANSITIONS_FILE
    data = np.load(path)
    obs = torch.from_numpy(data["obs"])
    action = torch.from_numpy(data["action"])
    reward = torch.from_numpy(data["reward"])
    done = torch.from_numpy(data["done"])
    next_obs = torch.from_numpy(data["next_obs"])
    # Datasets collected before the free-space head simply have no target for it.
    free = torch.from_numpy(data["free"]) if "free" in data.files else None

    # Keeping the whole dataset on the GPU removes the per-batch host copy. Only
    # worth it when it comfortably fits, hence the budget check.
    if device is not None and device.type == "cuda" and cfg.DATA_ON_DEVICE:
        gb = 2 * obs.numel() * obs.element_size() / 1e9
        if gb <= cfg.DATA_DEVICE_BUDGET_GB:
            try:
                obs, next_obs = obs.to(device), next_obs.to(device)
                action, reward, done = action.to(device), reward.to(device), done.to(device)
                free = free if free is None else free.to(device)
                print(f"[Data] dataset kept on {device} ({gb:.2f} GB)")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[Data] dataset too large for the GPU, streaming from RAM")

    n = len(action)
    n_val = int(n * cfg.VAL_FRACTION)
    n_train = n - n_val
    tr = Split(obs[:n_train], action[:n_train], reward[:n_train], done[:n_train],
               next_obs[:n_train], k, free=None if free is None else free[:n_train])
    va = Split(obs[n_train:], action[n_train:], reward[n_train:], done[n_train:],
               next_obs[n_train:], k, free=None if free is None else free[n_train:])
    tr.has_free = va.has_free = free is not None
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
def run_epoch(model, split, cfg, device, pos_weight, optimizer=None, space_coef=0.0):
    train = optimizer is not None
    model.train(train)
    # Accumulate on the device: reading a GPU tensor from Python (int(), item())
    # forces a sync, and doing that inside the K-step loop cost more than the
    # step itself.
    keys = ("loss", "head", "pred", "reward", "done", "space", "emb_std")
    agg = torch.zeros(len(keys), device=device)
    counts = torch.zeros(3, device=device)          # tp, fp, fn
    n_batches = 0

    for obs0, actions, rewards, dones, frees, next_seq in split.batches(
            cfg.BATCH_SIZE, shuffle=train, device=device):
        with torch.set_grad_enabled(train):
            s_pred = model.encode(obs0)
            std_loss, cov_loss, emb_std = vicreg_terms(s_pred)

            # Encode the whole K-step window in one call instead of K. Same FLOPs,
            # but this model is so small that kernel-launch latency dominated:
            # 10 encoder calls per batch cost more than the maths they did.
            b, k_steps = next_seq.shape[0], cfg.ROLLOUT_K
            flat = next_seq.reshape(b * k_steps, *next_seq.shape[2:])
            targets = model.encode_target(flat).view(b, k_steps, -1)
            trues = model.encode(flat).view(b, k_steps, -1)

            pred_l = rew_l = done_l = space_l = 0.0
            for k in range(cfg.ROLLOUT_K):
                a_k = actions[:, k]
                # True latent of the state the heads are supervised on.
                s_true = s_pred if k == 0 else trues[:, k - 1]
                # --- Heads supervised on the TRUE encoded latent (clean signal,
                #     including death states). Weighted MSE: rare food/death events
                #     count more than the many 0-reward steps.
                r_pred, r_tgt = model.reward(s_true, a_k), rewards[:, k]
                w = 1.0 + cfg.REWARD_EVENT_WEIGHT * (r_tgt.abs() > 0.5).float()
                rew_l = rew_l + (w * (r_pred - r_tgt) ** 2).mean()
                logit = model.done_logit(s_true, a_k)
                done_l = done_l + F.binary_cross_entropy_with_logits(
                    logit, dones[:, k], pos_weight=pos_weight)
                # Free space after the action, in [0, 1]: BCE on a soft target.
                space_l = space_l + F.binary_cross_entropy_with_logits(
                    model.space_logit(s_true, a_k), frees[:, k])

                pred_pos = logit > 0
                true_pos = dones[:, k] > 0.5
                counts += torch.stack([(pred_pos & true_pos).sum(),
                                       (pred_pos & ~true_pos).sum(),
                                       (~pred_pos & true_pos).sum()]).float()

                # --- Dynamics: predicted latent vs EMA target (stop-grad).
                s_pred = model.predict(s_pred, a_k)               # ŝ_{k+1}
                pred_l = pred_l + F.mse_loss(s_pred, targets[:, k])

            pred_l = pred_l / cfg.ROLLOUT_K
            rew_l = rew_l / cfg.ROLLOUT_K
            done_l = done_l / cfg.ROLLOUT_K
            space_l = space_l / cfg.ROLLOUT_K
            # Head loss (fixed-scale targets) drives model selection. The latent
            # prediction MSE is NOT scale-invariant (it grows as VICReg expands the
            # latent), so it must not drive selection - only reward/done do.
            head = (cfg.REWARD_COEF * rew_l + cfg.DONE_COEF * done_l
                    + space_coef * space_l)
            loss = (cfg.PRED_COEF * pred_l + cfg.STD_COEF * std_loss
                    + cfg.COV_COEF * cov_loss + head)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            model.ema_update(cfg.EMA_DECAY)

        agg += torch.stack([loss.detach(), head.detach(), pred_l.detach(),
                            rew_l.detach(), done_l.detach(), space_l.detach(),
                            emb_std])
        n_batches += 1

    # One single device -> host transfer, at the end of the epoch.
    values = (agg / max(n_batches, 1)).tolist()
    tp, fp, fn = counts.tolist()
    out = dict(zip(keys, values))
    out["done_recall"] = tp / max(tp + fn, 1.0)
    out["done_precision"] = tp / max(tp + fp, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(epochs: int | None = None, dataset=None, init_from=None,
         tag: str | None = None) -> dict:
    """Train the world model and return {model_path, best_val_head, ...}.

    ``init_from`` warm-starts from an existing checkpoint, which is what the
    DAgger loop needs: each round refines the same model on a dataset that now
    contains the states the planner actually visits.
    """
    cfg = config
    n_epochs = epochs if epochs is not None else cfg.EPOCHS
    set_seed(cfg.RANDOM_STATE)
    device = resolve_device(cfg.DEVICE)

    train_split, val_split, n_total = load_splits(cfg, cfg.ROLLOUT_K, path=dataset,
                                                  device=device)
    pos_weight = done_pos_weight(train_split, cfg.DONE_POS_WEIGHT_CAP, device)
    # No free-space target in the dataset -> the head cannot be trained.
    space_coef = cfg.SPACE_COEF if getattr(train_split, "has_free", False) else 0.0
    model = WorldModel(dim=cfg.EMBED_DIM, hidden=cfg.HIDDEN_DIM).to(device)
    if init_from is not None:
        model.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print(f"[Data] {n_total} transitions | train windows {len(train_split.starts)} "
          f"| val windows {len(val_split.starts)} | device {device} "
          f"| done pos_weight {float(pos_weight):.1f} | space_coef {space_coef}")

    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({
            "embed_dim": cfg.EMBED_DIM, "hidden_dim": cfg.HIDDEN_DIM,
            "lr": cfg.LEARNING_RATE, "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE, "epochs": n_epochs,
            "rollout_k": cfg.ROLLOUT_K, "ema_decay": cfg.EMA_DECAY,
            "pred_coef": cfg.PRED_COEF, "std_coef": cfg.STD_COEF,
            "cov_coef": cfg.COV_COEF, "reward_coef": cfg.REWARD_COEF,
            "done_coef": cfg.DONE_COEF, "done_pos_weight": float(pos_weight),
            "space_coef": space_coef, "mpc_horizon": cfg.MPC_HORIZON,
            "mpc_space_coef": cfg.MPC_SPACE_COEF, "env_max_steps": cfg.ENV_MAX_STEPS,
            "seed": cfg.RANDOM_STATE, "n_transitions": n_total,
            "warm_start": init_from is not None,
        })

        model_dir = next_run_dir(config.OUTPUTS_MODELS, cfg.MODEL_NAME)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_head = float("inf")   # select on the head loss (scale-stable), not the total
        bad_epochs = 0
        history = []

        for epoch in tqdm(range(n_epochs), desc="Training"):
            tr = run_epoch(model, train_split, cfg, device, pos_weight, optimizer,
                           space_coef=space_coef)
            va = run_epoch(model, val_split, cfg, device, pos_weight, optimizer=None,
                           space_coef=space_coef)

            mlflow.log_metrics({
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "train_head": tr["head"], "val_head": va["head"],
                "val_pred": va["pred"], "val_reward_mse": va["reward"],
                "val_space": va["space"],
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

    return {"model_path": model_dir / "best_model.pt", "best_val_head": best_head,
            "results_dir": results_dir, "history": history}


if __name__ == "__main__":
    main()
