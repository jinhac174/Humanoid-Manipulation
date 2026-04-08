"""
Observation space for the Insert task.

Dim = 124:
  [0:28]   joint_pos (28 actuated joints)
  [28:56]  joint_vel
  [56:84]  target_error = default_joint_pos - joint_pos
  [84:87]  block_pos  (robot-relative)
  [87:91]  block_quat (w,x,y,z) — orientation matters for insertion
  [91:94]  peg_a_pos  (robot-relative)
  [94:97]  peg_b_pos  (robot-relative)
  [97:100] left_palm_pos  (robot-relative)
  [100:103] right_palm_pos (robot-relative)
  [103:106] block_peg_a site (robot-relative)
  [106:109] block_peg_b site (robot-relative)
  [109:112] block_peg_a → peg_a  (insertion error vector)
  [112:115] block_peg_b → peg_b
  [115:118] left_palm → block
  [118:121] right_palm → block
  [121:124] block → left_palm (redundant but explicit inductive bias)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from .env import InsertEnv


def get_observations(env: "InsertEnv") -> torch.Tensor:
    robot      = env.robot
    block      = env.block
    peg_a      = env.peg_a
    peg_b      = env.peg_b
    robot_root = robot.data.root_pos_w  # (N,3) — robot origin for relative coords

    # ── Joint states ──────────────────────────────────────────────────────────
    joint_pos    = robot.data.joint_pos[:, env.actuated_joint_ids]
    joint_vel    = robot.data.joint_vel[:, env.actuated_joint_ids]
    target_error = robot.data.default_joint_pos[:, env.actuated_joint_ids] - joint_pos

    # ── Object positions (robot-relative) ─────────────────────────────────────
    block_pos_rel  = block.data.root_pos_w  - robot_root
    peg_a_pos_rel  = peg_a.data.root_pos_w  - robot_root
    peg_b_pos_rel  = peg_b.data.root_pos_w  - robot_root
    block_quat     = block.data.root_quat_w  # (N,4) keep in world frame

    # ── Hand positions (robot-relative) ──────────────────────────────────────
    left_palm_w  = robot.data.body_pos_w[:, env.left_palm_body_id,  :]
    right_palm_w = robot.data.body_pos_w[:, env.right_palm_body_id, :]
    left_palm_rel  = left_palm_w  - robot_root
    right_palm_rel = right_palm_w - robot_root

    # ── Block insertion sites (robot-relative) ────────────────────────────────
    block_pos_w      = block.data.root_pos_w
    block_peg_a_w    = block_pos_w + quat_apply(block_quat, env.block_peg_a_offset)
    block_peg_b_w    = block_pos_w + quat_apply(block_quat, env.block_peg_b_offset)
    block_peg_a_rel  = block_peg_a_w - robot_root
    block_peg_b_rel  = block_peg_b_w - robot_root

    # ── Insertion error vectors ───────────────────────────────────────────────
    peg_a_site = peg_a.data.root_pos_w + env.peg_site_offset
    peg_b_site = peg_b.data.root_pos_w + env.peg_site_offset

    block_peg_a_to_peg_a = peg_a_site - block_peg_a_w  # direction to insert
    block_peg_b_to_peg_b = peg_b_site - block_peg_b_w

    # ── Hand-to-object vectors ─────────────────────────────────────────────────
    left_palm_to_block  = block_pos_w - left_palm_w
    right_palm_to_block = block_pos_w - right_palm_w
    block_to_left_palm  = left_palm_w - block_pos_w

    return torch.cat([
        joint_pos,            # 28
        joint_vel,            # 28
        target_error,         # 28
        block_pos_rel,        # 3
        block_quat,           # 4
        peg_a_pos_rel,        # 3
        peg_b_pos_rel,        # 3
        left_palm_rel,        # 3
        right_palm_rel,       # 3
        block_peg_a_rel,      # 3
        block_peg_b_rel,      # 3
        block_peg_a_to_peg_a, # 3
        block_peg_b_to_peg_b, # 3
        left_palm_to_block,   # 3
        right_palm_to_block,  # 3
        block_to_left_palm,   # 3
    ], dim=-1)   # total: 28+28+28+3+4+3+3+3+3+3+3+3+3+3+3+3 = 124