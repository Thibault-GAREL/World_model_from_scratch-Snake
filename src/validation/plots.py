"""Generate the README figures from real training / evaluation artifacts.

Produces (into ``assets/``):
  - plot_training.png      training dynamics + collapse monitor
  - plot_drift.png         latent prediction error vs imagined horizon
  - plot_horizon.png       MPC score vs planning horizon (the drift trade-off)

Run:  python -m src.validation.plots
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import config, ROOT_DIR

ASSETS = ROOT_DIR / "assets"
INDIGO, EMERALD, ROSE, AMBER, CYAN = "#6366F1", "#10B981", "#F43F5E", "#F59E0B", "#06B6D4"


def _latest_history() -> dict:
    runs = sorted(config.OUTPUTS_RESULTS.glob(f"{config.MODEL_NAME}_run-*/metrics.json"))
    with open(runs[-1]) as f:
        return json.load(f)


def _latest_eval() -> dict:
    evals = sorted(config.OUTPUTS_RESULTS.glob("evaluation_*.json"))
    with open(evals[-1]) as f:
        return json.load(f)


def plot_training(history: list[dict]) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("JEPA world model - training dynamics", fontweight="bold", fontsize=14)

    ax[0, 0].plot(ep, [h["train_head"] for h in history], color=INDIGO, label="train")
    ax[0, 0].plot(ep, [h["val_head"] for h in history], color=ROSE, label="val")
    ax[0, 0].set_title("Head loss (reward + done) - model-selection metric")
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(ep, [h["train_emb_std"] for h in history], color=EMERALD)
    ax[0, 1].axhline(1.0, ls="--", color="gray", lw=1, label="VICReg target = 1.0")
    ax[0, 1].axhline(0.0, ls=":", color=ROSE, lw=1, label="collapse")
    ax[0, 1].set_title("Embedding std - COLLAPSE MONITOR (stays ~1 = healthy)")
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylim(-0.05, 1.2)
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(ep, [h["val_done_recall"] for h in history], color=INDIGO, label="recall")
    ax[1, 0].plot(ep, [h["val_done_precision"] for h in history], color=AMBER, label="precision")
    ax[1, 0].set_title("Death prediction - done head (deaths are ~2% of data)")
    ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylim(0, 1); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(ep, [h["val_reward"] for h in history], color=CYAN)
    ax[1, 1].set_title("Reward head loss (weighted MSE, val)")
    ax[1, 1].set_xlabel("epoch"); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "plot_training.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_drift(evaluation: dict) -> None:
    drift = evaluation["drift_per_horizon"]
    h = np.arange(1, len(drift) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(h, drift, "o-", color=INDIGO, lw=2, ms=7)
    ax.fill_between(h, 0, drift, color=INDIGO, alpha=0.08)
    ax.set_title("Compounding drift - latent prediction error grows with horizon",
                 fontweight="bold")
    ax.set_xlabel("imagined steps ahead (k)")
    ax.set_ylabel("latent MSE  ‖ŝ - target‖²")
    ax.set_xticks(h); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "plot_drift.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_horizon(evaluation: dict) -> None:
    """MPC score vs horizon (needs the model) + baselines from the eval json."""
    import torch
    from src.models.planner import LatentMPC, load_world_model, play_episode
    from src.models.train import resolve_device
    from src.envs.snake_env import SnakeEnv

    device = resolve_device(config.DEVICE)
    model = load_world_model(device)
    rng = np.random.default_rng(0)

    horizons = [1, 2, 3, 4, 5, 6, 7]
    means = []
    for H in horizons:
        planner = LatentMPC(model, horizon=H, gamma=config.MPC_GAMMA, device=device)
        env = SnakeEnv(max_steps=500, seed=2026)
        scores = [play_episode(env, lambda e, o: planner.act(o))["score"] for _ in range(25)]
        means.append(float(np.mean(scores)))

    greedy = evaluation["agent"]["greedy"]["mean_score"]
    rand = evaluation["agent"]["random"]["mean_score"]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(horizons, means, "o-", color=EMERALD, lw=2, ms=7, label="MPC (world model)")
    ax.axhline(greedy, ls="--", color=AMBER, lw=1.5, label=f"greedy heuristic ({greedy})")
    ax.axhline(rand, ls=":", color=ROSE, lw=1.5, label=f"random ({rand})")
    best = int(np.argmax(means))
    ax.annotate(f"optimum H={horizons[best]}",
                (horizons[best], means[best]),
                textcoords="offset points", xytext=(0, 12), ha="center",
                color=EMERALD, fontweight="bold")
    ax.set_title("Planning horizon vs score - drift caps the useful horizon",
                 fontweight="bold")
    ax.set_xlabel("MPC planning horizon H"); ax.set_ylabel("mean apples / episode")
    ax.set_xticks(horizons); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "plot_horizon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def latent_map(n: int = 4000) -> None:
    """Project the encoded latents to 2D (UMAP, t-SNE fallback) and colour them
    by snake length and distance-to-apple - shows the encoder organizes states
    by meaning, without ever being told to."""
    import torch
    from src.models.planner import load_world_model
    from src.models.train import resolve_device

    data = np.load(config.DATA_PROCESSED / config.TRANSITIONS_FILE)
    obs = data["obs"]
    idx = np.random.default_rng(0).choice(len(obs), size=min(n, len(obs)), replace=False)
    obs = obs[idx]

    device = resolve_device(config.DEVICE)
    model = load_world_model(device)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(obs).to(device)).cpu().numpy()

    length = obs[:, 1].reshape(len(obs), -1).sum(1)            # apples eaten (body cells)

    def manhattan(o):
        head = np.argwhere(o[0] > 0.5)
        apple = np.argwhere(o[2] > 0.5)
        if len(head) == 0 or len(apple) == 0:
            return np.nan
        return abs(head[0][0] - apple[0][0]) + abs(head[0][1] - apple[0][1])
    dist = np.array([manhattan(o) for o in obs])

    try:
        import umap
        proj = umap.UMAP(n_neighbors=25, min_dist=0.12, random_state=0).fit_transform(z)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, random_state=0, init="pca",
                    perplexity=30).fit_transform(z)
        method = "t-SNE"

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Latent space ({method}) - the encoder self-organizes states by meaning",
                 fontweight="bold", fontsize=14)
    sc0 = ax[0].scatter(proj[:, 0], proj[:, 1], c=length, cmap="viridis", s=7, alpha=0.8)
    ax[0].set_title("coloured by snake length (apples eaten)")
    fig.colorbar(sc0, ax=ax[0], shrink=0.85, label="apples eaten")
    sc1 = ax[1].scatter(proj[:, 0], proj[:, 1], c=dist, cmap="magma", s=7, alpha=0.8)
    ax[1].set_title("coloured by Manhattan distance head → apple")
    fig.colorbar(sc1, ax=ax[1], shrink=0.85, label="distance to apple")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "latent_umap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"latent map ({method}) saved")


def render_gameplay_gif(n_try: int = 10) -> None:
    """Record the MPC agent playing one episode as a GIF (grid animation)."""
    from matplotlib import animation
    from src.models.planner import LatentMPC, load_world_model
    from src.models.train import resolve_device
    from src.envs.snake_env import SnakeEnv, GRID_H, GRID_W

    device = resolve_device(config.DEVICE)
    model = load_world_model(device)
    planner = LatentMPC(model, horizon=config.MPC_HORIZON, gamma=config.MPC_GAMMA, device=device)

    best_frames, best_score = None, -1
    for seed in range(n_try):
        env = SnakeEnv(max_steps=500, seed=1000 + seed)
        obs = env.reset()
        frames, done, info = [obs.copy()], False, {"score": 0}
        while not done:
            obs, _, done, info = env.step(planner.act(obs))
            frames.append(obs.copy())
        if info["score"] > best_score:
            best_score, best_frames = info["score"], frames

    def to_rgb(o):
        img = np.full((GRID_H, GRID_W, 3), (0.16, 0.16, 0.24))
        img[o[1] > 0.5] = (0.0, 0.78, 0.55)    # body (teal)
        img[o[0] > 0.5] = (0.20, 1.0, 0.40)    # head (green)
        img[o[2] > 0.5] = (0.96, 0.26, 0.36)   # apple (red)
        return img

    fig, ax = plt.subplots(figsize=(6, 3.2)); ax.axis("off")
    im = ax.imshow(to_rgb(best_frames[0]), interpolation="nearest")
    ttl = ax.set_title("", fontsize=11, fontweight="bold")

    def update(i):
        im.set_data(to_rgb(best_frames[i]))
        ttl.set_text(f"JEPA world model · MPC (no policy) · apples: {int(best_frames[i][1].sum())}")
        return im, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(best_frames), interval=110, blit=False)
    anim.save(ASSETS / "mpc_gameplay.gif", writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print(f"gameplay gif: best score {best_score} over {len(best_frames)} frames")


def render_pygame_gif(n_try: int = 8, max_steps: int = 280) -> None:
    """Record the agent playing inside the ORIGINAL pygame renderer (snake.py).

    The planner drives the real game engine; we capture the styled snake (eyes,
    body) frame by frame, headless via the SDL 'dummy' driver.
    """
    import os
    import random
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    import torch
    from PIL import Image
    import snake as game
    from src.models.planner import LatentMPC, load_world_model
    from src.models.train import resolve_device

    C = 50  # snake.py cell size (px)
    device = resolve_device(config.DEVICE)
    model = load_world_model(device)
    planner = LatentMPC(model, horizon=config.MPC_HORIZON, gamma=config.MPC_GAMMA, device=device)

    def grid_obs(mgr, fd):
        o = np.zeros((3, 8, 16), dtype=np.float32)
        h = mgr.list_snake[0]
        o[0, h.y // C, h.x // C] = 1.0
        for s in mgr.list_snake[1:]:
            o[1, s.y // C, s.x // C] = 1.0
        o[2, fd.y // C, fd.x // C] = 1.0
        return o

    best_frames, best_score = None, -1
    for trial in range(n_try):
        random.seed(100 + trial)
        mgr = game.Manager_snake()
        mgr.add_snake(game.Snake(5 * C, 5 * C))
        fd = game.generated_food(mgr)
        frames, score = [], 0
        for _ in range(max_steps):
            a = planner.act(grid_obs(mgr, fd))
            if a == 0 and mgr.direction != "DOWN":   mgr.direction = "UP"
            elif a == 2 and mgr.direction != "UP":   mgr.direction = "DOWN"
            elif a == 1 and mgr.direction != "LEFT": mgr.direction = "RIGHT"
            elif a == 3 and mgr.direction != "RIGHT": mgr.direction = "LEFT"
            if mgr.list_snake[0].x == fd.x and mgr.list_snake[0].y == fd.y:
                mgr.add_snake(game.Snake(mgr.list_snake[-1].x, mgr.list_snake[-1].y))
                fd = game.generated_food(mgr)
                score += 1
            if mgr.move() is False:
                break
            game.display.fill(game.BLACK)
            pygame.draw.rect(game.display, game.RED, (fd.x, fd.y, C, C))
            mgr.draw_snake()
            frames.append(pygame.surfarray.array3d(game.display).swapaxes(0, 1).copy())
        if score > best_score:
            best_score, best_frames = score, frames

    imgs = [Image.fromarray(f).resize((560, 280), Image.NEAREST) for f in best_frames]
    imgs[0].save(ASSETS / "snake_gameplay.gif", save_all=True,
                 append_images=imgs[1:], duration=90, loop=0)
    print(f"pygame gif: best score {best_score} over {len(best_frames)} frames")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    history = _latest_history()["history"]
    evaluation = _latest_eval()
    plot_training(history)
    plot_drift(evaluation)
    plot_horizon(evaluation)
    latent_map()
    render_gameplay_gif()       # raw-grid view
    render_pygame_gif()         # original pygame engine
    print(f"Saved figures + gifs -> {ASSETS}")


if __name__ == "__main__":
    main()
