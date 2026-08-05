"""Architecture du modèle (sans logique d'entraînement).

Action-conditioned JEPA world model:

    encoder        : grid observation (3x8x16) -> latent s_t  (dim D)
    target_encoder : EMA copy of the encoder (no gradient) -> learning target
    predictor      : (s_t, a_t) -> ŝ_{t+1}   (residual, in latent space)
    reward_head    : (s_t, a_t) -> r̂
    done_head      : (s_t, a_t) -> done logit
    space_head     : (s_t, a_t) -> fraction of the board still reachable

The space head is what lets a short-horizon planner avoid trapping itself. The
done head only sees deaths inside the horizon, so with H=5 the agent happily
walks into a pocket it will die in 20 steps later. The free-space head turns
that long-term consequence into a signal readable at the current step.

The JEPA loss, VICReg and the EMA schedule live in ``train.py``, this file only
defines the modules and the elementary operations (encode / predict / EMA copy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.snake_env import GRID_H, GRID_W, N_CHANNELS, N_ACTIONS


class Encoder(nn.Module):
    """Small CNN: (B, 3, 8, 16) -> (B, dim)."""

    def __init__(self, in_ch: int = N_CHANNELS, dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * GRID_H * GRID_W, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class WorldModel(nn.Module):
    """JEPA world model with reward / done heads and an EMA target encoder."""

    def __init__(self, dim: int = 128, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions

        self.encoder = Encoder(dim=dim)
        self.target_encoder = Encoder(dim=dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = _mlp(dim + n_actions, hidden, dim)
        self.reward_head = _mlp(dim + n_actions, hidden, 1)
        self.done_head = _mlp(dim + n_actions, hidden, 1)
        self.space_head = _mlp(dim + n_actions, hidden, 1)
        self._ema_pairs = None        # lazily cached parameter lists for the EMA

    # --- elementary ops -------------------------------------------------
    def _cat_action(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a, self.n_actions).to(s.dtype)
        return torch.cat([s, a_onehot], dim=-1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(obs)

    def predict(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict the next latent (residual update in latent space)."""
        return s + self.predictor(self._cat_action(s, a))

    def reward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward_head(self._cat_action(s, a)).squeeze(-1)

    def done_logit(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.done_head(self._cat_action(s, a)).squeeze(-1)

    def space_logit(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.space_head(self._cat_action(s, a)).squeeze(-1)

    def space(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predicted fraction of the board still reachable after the action."""
        return torch.sigmoid(self.space_logit(s, a))

    @torch.no_grad()
    def ema_update(self, tau: float = 0.99) -> None:
        """Move the target encoder toward the online encoder: θ_t ← τ θ_t + (1−τ) θ.

        Fused over the parameter list: on a model this small the per-tensor loop
        was launching more kernels than the forward pass itself.
        """
        if self._ema_pairs is None:
            self._ema_pairs = (list(self.target_encoder.parameters()),
                               list(self.encoder.parameters()))
        target_params, online_params = self._ema_pairs
        torch._foreach_mul_(target_params, tau)
        torch._foreach_add_(target_params, online_params, alpha=1.0 - tau)
        for tb, b in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            tb.copy_(b)
