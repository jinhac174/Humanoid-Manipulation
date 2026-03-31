import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform


def reset_robot(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    env.nominal_joint_pos[env_ids] = joint_pos[:, env.actuated_joint_ids]
    env.target_joint_pos[env_ids]  = env.nominal_joint_pos[env_ids]
    env.actions[env_ids]           = 0.0
    env.prev_actions[env_ids]      = 0.0


def reset_can(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    can_state = env.can.data.default_root_state[env_ids].clone()

    can_state[:, 0] += sample_uniform(
        env.cfg.can_spawn_x_range[0], env.cfg.can_spawn_x_range[1], (n,), device=env.device
    )
    can_state[:, 1] += sample_uniform(
        env.cfg.can_spawn_y_range[0], env.cfg.can_spawn_y_range[1], (n,), device=env.device
    )

    yaw   = sample_uniform(0.0, 2 * torch.pi, (n,), device=env.device)
    zeros = torch.zeros_like(yaw)
    can_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    can_state[:, 7:]  = 0.0

    env.can.write_root_state_to_sim(can_state, env_ids=env_ids)
    env.can_spawn_z[env_ids] = can_state[:, 2]


def reset_buffers(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    left_palm = env.robot.data.body_pos_w[env_ids, env.left_palm_idx, :]
    can_pos   = env.can.data.root_pos_w[env_ids]

    env.prev_left_dist[env_ids] = torch.norm(left_palm - can_pos, dim=-1)
    env.prev_can_to_target[env_ids] = torch.norm(
        can_pos[:, :2] - env.target_pos_w[env_ids, :2], dim=-1
    )