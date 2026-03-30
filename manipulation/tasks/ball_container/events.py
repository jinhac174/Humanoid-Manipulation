import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform


def reset_robot(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    """Reset robot joints to nominal pose with zero velocity."""
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    env.nominal_joint_pos[env_ids] = joint_pos[:, env.actuated_joint_ids]
    env.target_joint_pos[env_ids]  = env.nominal_joint_pos[env_ids]
    env.actions[env_ids]           = 0.0
    env.prev_actions[env_ids]      = 0.0


def reset_ball(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    """
    Reset ball to default spawn position with XY randomization and random yaw.
    Ranges read from scene config.
    """
    n = len(env_ids)
    ball_state = env.ball.data.default_root_state[env_ids].clone()

    x_range = env.cfg.scene.ball_spawn_x_range
    y_range = env.cfg.scene.ball_spawn_y_range

    ball_state[:, 0] += sample_uniform(x_range[0], x_range[1], (n,), device=env.device)
    ball_state[:, 1] += sample_uniform(y_range[0], y_range[1], (n,), device=env.device)

    yaw   = sample_uniform(0.0, 2 * torch.pi, (n,), device=env.device)
    zeros = torch.zeros_like(yaw)
    ball_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    ball_state[:, 7:]  = 0.0

    env.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
    env.ball_spawn_z[env_ids] = ball_state[:, 2]


def reset_container(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    """Reset container to fixed default position with zero velocity."""
    container_state = env.container.data.default_root_state[env_ids].clone()
    container_state[:, 7:] = 0.0
    env.container.write_root_state_to_sim(container_state, env_ids=env_ids)
    env.container_spawn_z[env_ids] = container_state[:, 2]


def reset_buffers(env: DirectRLEnv, env_ids: torch.Tensor) -> None:
    """Reset per-env progress tracking buffers."""
    left_palm  = env.robot.data.body_pos_w[env_ids, env.left_palm_idx, :]
    ball_pos   = env.ball.data.root_pos_w[env_ids]
    container_pos = env.container.data.root_pos_w[env_ids]

    env.prev_left_dist[env_ids] = torch.norm(
        left_palm - ball_pos, dim=-1
    )
    env.prev_ball_height[env_ids] = (
        ball_pos[:, 2] - env.ball_spawn_z[env_ids]
    )
    env.prev_ball_to_container[env_ids] = torch.norm(
        ball_pos[:, :2] - container_pos[:, :2], dim=-1
    )