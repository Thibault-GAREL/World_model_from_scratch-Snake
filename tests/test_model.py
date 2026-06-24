"""Forward-pass and EMA shape checks for the JEPA world model."""

import torch

from src.envs.snake_env import GRID_H, GRID_W, N_CHANNELS
from src.models.model import WorldModel


def _batch(b: int = 8):
    obs = torch.rand(b, N_CHANNELS, GRID_H, GRID_W)
    actions = torch.randint(0, 4, (b,))
    return obs, actions


def test_forward_shapes():
    model = WorldModel(dim=128)
    obs, a = _batch(8)

    s = model.encode(obs)
    assert s.shape == (8, 128)

    s_next = model.predict(s, a)
    assert s_next.shape == (8, 128)

    assert model.reward(s, a).shape == (8,)
    assert model.done_logit(s, a).shape == (8,)


def test_target_encoder_is_detached():
    model = WorldModel()
    obs, _ = _batch(4)
    s_target = model.encode_target(obs)
    assert not s_target.requires_grad
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_ema_update_moves_target_toward_online():
    model = WorldModel()
    # Perturb the online encoder so it differs from the (initially identical) target.
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p))

    p_online = next(model.encoder.parameters()).clone()
    p_target_before = next(model.target_encoder.parameters()).clone()
    model.ema_update(tau=0.9)
    p_target_after = next(model.target_encoder.parameters())

    # Target moved a bit toward the online weights, but not all the way.
    assert torch.norm(p_target_after - p_target_before) > 0
    assert torch.norm(p_target_after - p_online) < torch.norm(p_target_before - p_online)
