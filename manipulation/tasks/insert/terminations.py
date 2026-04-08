"""
Termination conditions for the Insert task.
Ported from HumanoidBench insert.py — robot-fall check removed (fixed-base).
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .env import InsertEnv

from .env_cfg import BLOCK_PEG_A_OFFSET, BLOCK_PEG_B_OFFSET, OBJECT_DROP_Z


def object_dropped(env: "InsertEnv") -> torch.Tensor:
    """
    Any of: block_peg_a, block_peg_b, peg_a, peg_b fall below OBJECT_DROP_Z.
    HumanoidBench: terminate if site_z < 0.5
    """
    from isaaclab.utils.math import quat_apply

    block_pos  = env.block.data.root_pos_w
    block_quat = env.block.data.root_quat_w

    block_peg_a_z = (block_pos + quat_apply(block_quat, env.block_peg_a_offset))[:, 2]
    block_peg_b_z = (block_pos + quat_apply(block_quat, env.block_peg_b_offset))[:, 2]

    peg_a_z = env.peg_a.data.root_pos_w[:, 2]
    peg_b_z = env.peg_b.data.root_pos_w[:, 2]

    threshold = torch.full((env.num_envs,), OBJECT_DROP_Z, device=env.device)

    dropped = (
        (block_peg_a_z < threshold) |
        (block_peg_b_z < threshold) |
        (peg_a_z       < threshold) |
        (peg_b_z       < threshold)
    )
    return dropped


def success(env: "InsertEnv") -> torch.Tensor:
    """
    Block insertion sites aligned with peg sites within 0.05m (5cm tolerance).
    """
    from isaaclab.utils.math import quat_apply

    block_pos  = env.block.data.root_pos_w
    block_quat = env.block.data.root_quat_w

    block_peg_a_w = block_pos + quat_apply(block_quat, env.block_peg_a_offset)
    block_peg_b_w = block_pos + quat_apply(block_quat, env.block_peg_b_offset)

    peg_a_site = env.peg_a.data.root_pos_w + env.peg_site_offset
    peg_b_site = env.peg_b.data.root_pos_w + env.peg_site_offset

    dist_a = torch.norm(block_peg_a_w - peg_a_site, dim=-1)
    dist_b = torch.norm(block_peg_b_w - peg_b_site, dim=-1)

    return (dist_a < 0.05) & (dist_b < 0.05)