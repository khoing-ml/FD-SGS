"""Model-independent operations, using float32 latent-space geometry."""

import math

import torch


def rms(x):
    return x.float().flatten(1).square().mean(1).sqrt().reshape(
        (-1,) + (1,) * (x.ndim - 1)
    )


def fd_direction(anchor_clean, twin_clean, anchor_reward, twin_reward, eps=1e-8):
    delta = twin_clean.float() - anchor_clean.float()
    reward_delta = (twin_reward - anchor_reward).to(delta).reshape(
        (-1,) + (1,) * (delta.ndim - 1)
    )
    return reward_delta * delta / (rms(delta) + eps)


def stein_field(states, directions, repulsion=0.1, eps=1e-8):
    """RBF kernel exp(-||xi-xj||²/h), median/log(K+1) bandwidth.

    Each row i receives sum_j k(xj,xi) gj + gamma grad_xj k(xi,xj).
    Only anchors for ONE prompt may be passed in a call.
    """
    if states.shape != directions.shape:
        raise ValueError("States and directions must have identical shapes")
    k = len(states)
    if k == 1:
        return directions.float()
    x, g = states.float().flatten(1), directions.float().flatten(1)
    dist2 = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist").square()
    positive = dist2[dist2 > 0]
    h = (positive.median() / math.log(k + 1)).clamp_min(eps) if positive.numel() else x.new_tensor(1.)
    kernel = torch.exp(-dist2 / h)
    # Positive xi-xj is essential: the source gradient pushes targets apart.
    repel = (2 / h) * (x * kernel.sum(1, keepdim=True) - kernel @ x)
    return ((kernel @ g + repulsion * repel) / k).reshape_as(states)


def relative_correction(field, base_displacement, rho=0.025, trust_ratio=0.1, eps=1e-8):
    """Relative normalization with an explicit per-particle trust-region cap."""
    ratio = min(rho, trust_ratio)
    return ratio * rms(base_displacement) * field / (rms(field) + eps)


def stochastic_flow_step(state, velocity, sigma, sigma_next, gamma, noise):
    """FDFO's flow-adapted EDM overshoot/re-noise step at sigma_next.

    For x_s=(1-s)*data+s*noise, preserve the data coefficient by scaling
    the overshot sample by (1-s_next)/(1-s_tilde), then restore noise variance.
    velocity is dx/dsigma (negative of Z-Image transformer output).
    """
    s, target = float(sigma), float(sigma_next)
    if not 0 <= target <= s <= 1 or gamma < 0:
        raise ValueError("Require 0 <= sigma_next <= sigma <= 1 and gamma >= 0")
    if gamma == 0 or target == 0:
        return state + (target - s) * velocity
    overshoot = target / (1 + gamma * (1 - target))
    scale = (1 - target) / (1 - overshoot)
    std = math.sqrt(max(0., target**2 - (scale * overshoot)**2))
    return scale * (state + (overshoot - s) * velocity) + std * noise


def flow_grpo_step(state, velocity, sigma, sigma_next, noise_level, noise, sigma_cap):
    """Flow-GRPO Euler–Maruyama transition in decreasing noise time.

    Implements the `sde` transition (not CPS) from yifan123/flow_grpo,
    sd3_sde_with_logprob.py. sigma_cap is the schedule's second sigma,
    used only to regularize the denominator at sigma=1, as upstream does.
    Sampling only: no transition log-probabilities or RL objective.
    """
    s, target = float(sigma), float(sigma_next)
    if not 0 <= target <= s <= 1:
        raise ValueError("Require 0 <= sigma_next <= sigma <= 1")
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative")
    state, velocity = state.float(), velocity.float()
    dt = target - s
    if noise_level == 0 or dt == 0:
        return state + dt * velocity
    denominator_sigma = float(sigma_cap) if s == 1 else s
    if not 0 <= denominator_sigma < 1:
        raise ValueError("sigma_cap must be in [0, 1) at sigma=1")
    diffusion_squared = noise_level**2 * s / (1 - denominator_sigma)
    # A decreasing-time SDE needs the drift correction as well as fresh noise.
    drift = velocity + diffusion_squared / (2 * s) * (state + (1 - s) * velocity)
    return state + dt * drift + math.sqrt(diffusion_squared * -dt) * noise.float()
