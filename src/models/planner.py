"""Latent-space MPC controller: the world model acts by planning.

No learned policy. At each step the controller encodes the observation, imagines
every action sequence of length ``horizon`` inside the latent world model, scores
each imagined future with the reward / done heads, and plays the first action of
the best sequence (receding-horizon MPC).

Run:  python -m src.models.planner
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from src.config import config
from src.envs.snake_env import SnakeEnv, N_ACTIONS
from src.models.model import WorldModel


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def find_latest_model(model_name: str | None = None):
    """Return the best_model.pt of the highest-numbered run."""
    model_name = model_name or config.MODEL_NAME
    candidates = sorted(config.OUTPUTS_MODELS.glob(f"{model_name}_run-*/best_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model under {config.OUTPUTS_MODELS}. Run training first.")
    return candidates[-1]


def load_world_model(device: torch.device, path=None) -> WorldModel:
    path = path or find_latest_model()
    model = WorldModel(dim=config.EMBED_DIM, hidden=config.HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


class LatentMPC:
    """Receding-horizon planner over imagined latent rollouts."""

    def __init__(self, model: WorldModel, horizon: int = 4, gamma: float = 0.99,
                 device: torch.device | str = "cpu"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.gamma = gamma
        self.horizon = horizon
        # All action sequences of length `horizon` (M = N_ACTIONS ** horizon).
        seqs = list(itertools.product(range(N_ACTIONS), repeat=horizon))
        self.seqs = torch.tensor(seqs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = self.model.encode(x)                       # (1, D)
        m = self.seqs.shape[0]
        s = s.expand(m, -1).contiguous()               # (M, D)

        alive = torch.ones(m, device=self.device)      # cumulative survival prob
        ret = torch.zeros(m, device=self.device)       # expected discounted return
        g = 1.0
        for k in range(self.horizon):
            a_k = self.seqs[:, k]
            r = self.model.reward(s, a_k)
            d = torch.sigmoid(self.model.done_logit(s, a_k))
            ret = ret + g * alive * r                  # death reward (-1) is included via r
            alive = alive * (1.0 - d)                  # discount future by survival prob
            s = self.model.predict(s, a_k)
            g *= self.gamma

        best = int(torch.argmax(ret))
        return int(self.seqs[best, 0])


def play_episode(env: SnakeEnv, policy, max_steps: int = 500) -> dict:
    """Run one episode; ``policy`` is a callable (env, obs) -> action."""
    obs = env.reset()
    total_r, info = 0.0, {"score": 0, "steps": 0}
    for _ in range(max_steps):
        obs, r, done, info = env.step(policy(env, obs))
        total_r += r
        if done:
            break
    return {"score": info["score"], "steps": info["steps"], "return": total_r}


def main(episodes: int = 30) -> None:
    device = resolve_device(config.DEVICE)
    model = load_world_model(device)
    planner = LatentMPC(model, horizon=config.MPC_HORIZON,
                        gamma=config.MPC_GAMMA, device=device)

    env = SnakeEnv(max_steps=500, seed=2026)
    scores = [play_episode(env, lambda e, o: planner.act(o))["score"]
              for _ in range(episodes)]
    print(f"[MPC] horizon={config.MPC_HORIZON} | {episodes} episodes "
          f"| mean score {np.mean(scores):.2f} | max {int(np.max(scores))}")


if __name__ == "__main__":
    main()
