"""
Reward functions for the Insert task — fixed-base G1.

Adapted from HumanoidBench insert.py for fixed-base PPO.

HumanoidBench original:
    reward = (0.5*(small_control*stand_reward) + 0.5*cube_target)
             * (0.5*peg_height + 0.5*hand_proximity)

The stand_reward term served as an anti-stillness gate — without it,
small_control gives ~1.0 for T-pose and creates a stable local optimum.

Fixed-base adaptation:
    Stage 1 (always on):  hand_block_proximity  — approach the block
    Stage 2 (gated):      hand_block * cube_target * peg_height
    Smoothness:           -0.01 * action_rate

Doing nothing gives ≈ 0.015 reward.
Moving hands to block gives ≈ 0.30 reward.
Full insertion gives ≈ 1.0 reward.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from .env import InsertEnv


# ── Tolerance (dm_control port) ───────────────────────────────────────────────

def _linear_sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    scale    = 1.0 - value_at_1
    scaled_x = x * scale
    return torch.where(torch.abs(scaled_x) < 1.0, 1.0 - scaled_x, torch.zeros_like(x))


def _gaussian_sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    scale = math.sqrt(-2.0 * math.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def _quadratic_sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
    scale    = math.sqrt(1.0 - value_at_1)
    scaled_x = x * scale
    return torch.where(torch.abs(scaled_x) < 1.0, 1.0 - scaled_x ** 2, torch.zeros_like(x))


def tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    lower, upper = bounds
    in_bounds = (x >= lower) & (x <= upper)
    if margin == 0.0:
        return in_bounds.float()
    d = torch.where(x < lower, lower - x, x - upper) / margin
    if sigmoid == "gaussian":
        sig = _gaussian_sigmoid(d, value_at_margin)
    elif sigmoid == "linear":
        sig = _linear_sigmoid(d, value_at_margin)
    elif sigmoid == "quadratic":
        sig = _quadratic_sigmoid(d, value_at_margin)
    else:
        raise ValueError(f"Unknown sigmoid: {sigmoid!r}")
    return torch.where(in_bounds, torch.ones_like(x), sig)


# ── Reward terms ──────────────────────────────────────────────────────────────

def hand_block_proximity_reward(env: "InsertEnv") -> torch.Tensor:
    """
    Stage 1: at least one hand approaches the block.
    Replaces stand_reward as the anti-stillness gate.
    Doing nothing → hands ~0.5m from block → reward ≈ 0.05.
    """
    left_palm  = env.robot.data.body_pos_w[:, env.left_palm_body_id,  :]
    right_palm = env.robot.data.body_pos_w[:, env.right_palm_body_id, :]
    block_pos  = env.block.data.root_pos_w

    left_dist  = torch.norm(left_palm  - block_pos, dim=-1)
    right_dist = torch.norm(right_palm - block_pos, dim=-1)
    min_dist   = torch.minimum(left_dist, right_dist)

    return tolerance(min_dist, bounds=(0.0, 0.15), margin=0.8)


def cube_target_reward(env: "InsertEnv") -> torch.Tensor:
    """
    Block insertion sites align with peg sites.
    From HumanoidBench: tolerance(dist, margin=0.5, linear) averaged over a/b.
    """
    block_pos  = env.block.data.root_pos_w
    block_quat = env.block.data.root_quat_w
    peg_a_pos  = env.peg_a.data.root_pos_w
    peg_b_pos  = env.peg_b.data.root_pos_w

    block_peg_a_w = block_pos + quat_apply(block_quat, env.block_peg_a_offset)
    block_peg_b_w = block_pos + quat_apply(block_quat, env.block_peg_b_offset)

    peg_a_site = peg_a_pos + env.peg_site_offset
    peg_b_site = peg_b_pos + env.peg_site_offset

    dist_a = torch.norm(block_peg_a_w - peg_a_site, dim=-1)
    dist_b = torch.norm(block_peg_b_w - peg_b_site, dim=-1)

    r_a = tolerance(dist_a, margin=0.5, sigmoid="linear")
    r_b = tolerance(dist_b, margin=0.5, sigmoid="linear")
    return (r_a + r_b) * 0.5


def peg_height_reward(env: "InsertEnv") -> torch.Tensor:
    """
    Pegs stay on table.
    From HumanoidBench: tolerance(peg_z - target, margin=0.15, linear).
    """
    peg_a_z = env.peg_a.data.root_pos_w[:, 2] + env.peg_site_offset[0, 2]
    peg_b_z = env.peg_b.data.root_pos_w[:, 2] + env.peg_site_offset[0, 2]
    target   = env.cfg.peg_height_target

    r_a = tolerance(peg_a_z - target, margin=0.15, sigmoid="linear")
    r_b = tolerance(peg_b_z - target, margin=0.15, sigmoid="linear")
    return (r_a + r_b) * 0.5


def action_rate_penalty(env: "InsertEnv") -> torch.Tensor:
    """Penalize jerky actions — smoothness term."""
    return torch.norm(env.actions - env.prev_actions, dim=-1)


def compute_reward(env: "InsertEnv") -> tuple[torch.Tensor, dict]:
    """
    reward = 0.3 * hand_block_proximity
           + 0.7 * hand_block_proximity * cube_target * peg_height
           - 0.01 * action_rate

    hand_block_proximity gates the insertion reward — robot must
    approach the block before getting credit for alignment.
    """
    r_hand_block = hand_block_proximity_reward(env)
    r_cube       = cube_target_reward(env)
    r_height     = peg_height_reward(env)
    r_action     = action_rate_penalty(env)

    env.prev_actions = env.actions.clone()

    reward = (
        0.3 * r_hand_block
        + 0.7 * r_hand_block * r_cube * r_height
    ) * env.cfg.reward_scale

    info = {
        "hand_block_proximity": r_hand_block,
        "cube_target_reward":   r_cube,
        "peg_height_reward":    r_height,
        "action_rate":          r_action,
    }

    if env.common_step_counter % 100 == 0:
        print(f"hand_block={r_hand_block.mean():.3f} cube={r_cube.mean():.3f} height={r_height.mean():.3f} action={r_action.mean():.3f}")

    return reward, info