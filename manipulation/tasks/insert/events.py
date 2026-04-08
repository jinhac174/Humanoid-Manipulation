"""
Reset events for the Insert task.
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_from_euler_xyz

from .env_cfg import (
    PEG_A_INIT_POS, PEG_B_INIT_POS, BLOCK_INIT_POS,
)

if TYPE_CHECKING := False:
    from .env import InsertEnv


def reset_robot(env, env_ids: torch.Tensor) -> None:
    """Reset robot to default joint positions with small Gaussian noise."""
    robot = env.robot
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos += torch.randn_like(joint_pos) * 0.02
    joint_pos = torch.clamp(
        joint_pos,
        robot.data.soft_joint_pos_limits[env_ids, :, 0],
        robot.data.soft_joint_pos_limits[env_ids, :, 1],
    )
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_pose_to_sim(robot.data.default_root_state[env_ids, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def reset_objects(env, env_ids: torch.Tensor) -> None:
    """Reset peg_a, peg_b, block to init positions with small XY randomization."""
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]  # (n,3)

    def _reset_rigid(obj, base_pos, x_range, y_range, yaw_range=None):
        pos = torch.tensor(base_pos, device=env.device).unsqueeze(0).expand(n, -1).clone()
        pos[:, 0] += torch.zeros(n, device=env.device).uniform_(*x_range)
        pos[:, 1] += torch.zeros(n, device=env.device).uniform_(*y_range)
        pos_world = pos + origins

        if yaw_range is not None:
            yaw = torch.zeros(n, device=env.device).uniform_(*yaw_range)
            zero = torch.zeros(n, device=env.device)
            quat = quat_from_euler_xyz(zero, zero, yaw)  # (n,4) w,x,y,z
        else:
            quat = torch.zeros(n, 4, device=env.device)
            quat[:, 0] = 1.0  # identity

        root_state = torch.cat([pos_world, quat], dim=-1)
        obj.write_root_pose_to_sim(root_state, env_ids=env_ids)
        obj.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )

    cfg = env.cfg
    _reset_rigid(env.peg_a,  PEG_A_INIT_POS,  cfg.peg_spawn_x_range,   cfg.peg_spawn_y_range)
    _reset_rigid(env.peg_b,  PEG_B_INIT_POS,  cfg.peg_spawn_x_range,   cfg.peg_spawn_y_range)
    _reset_rigid(env.block,  BLOCK_INIT_POS,  cfg.block_spawn_x_range, cfg.block_spawn_y_range,
                 yaw_range=cfg.block_spawn_yaw_range)


def reset_buffers(env, env_ids: torch.Tensor) -> None:
    """Reset per-env reward/info buffers."""
    env.episode_reward[env_ids] = 0.0
    env.success_buf[env_ids]    = False