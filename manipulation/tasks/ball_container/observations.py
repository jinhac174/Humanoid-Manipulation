import torch
from isaaclab.envs import DirectRLEnv


def joint_pos(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.robot.data.joint_pos[:, joint_ids]


def joint_vel(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.robot.data.joint_vel[:, joint_ids]


def target_error(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.target_joint_pos - env.robot.data.joint_pos[:, joint_ids]


def ball_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return env.ball.data.root_pos_w - env.robot.data.root_pos_w


def ball_lin_vel(env: DirectRLEnv) -> torch.Tensor:
    return env.ball.data.root_lin_vel_w


def container_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return env.container.data.root_pos_w - env.robot.data.root_pos_w


def left_palm_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return (env.robot.data.body_pos_w[:, env.left_palm_idx, :] - env.robot.data.root_pos_w)


def right_palm_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return ( env.robot.data.body_pos_w[:, env.right_palm_idx, :] - env.robot.data.root_pos_w)


def ball_to_left_palm(env: DirectRLEnv) -> torch.Tensor:
    return (env.robot.data.body_pos_w[:, env.left_palm_idx, :] - env.ball.data.root_pos_w)


def ball_to_right_palm(env: DirectRLEnv) -> torch.Tensor:
    return (env.robot.data.body_pos_w[:, env.right_palm_idx, :] - env.ball.data.root_pos_w)


def ball_to_container(env: DirectRLEnv) -> torch.Tensor:
    return env.container.data.root_pos_w - env.ball.data.root_pos_w


def get_obs(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    """
    108-dim observation vector.

    Layout:
        [0:28]    joint_pos             — actuated joint positions
        [28:56]   joint_vel             — actuated joint velocities
        [56:84]   target_error          — target_joint_pos - joint_pos
        [84:87]   ball_pos              — relative to robot root
        [87:90]   ball_lin_vel          — world frame
        [90:93]   container_pos         — relative to robot root
        [93:96]   left_palm_pos         — relative to robot root
        [96:99]   right_palm_pos        — relative to robot root
        [99:102]  ball_to_left_palm     — left_palm_pos_w - ball_pos_w
        [102:105] ball_to_right_palm    — right_palm_pos_w - ball_pos_w
        [105:108] ball_to_container     — container_pos_w - ball_pos_w
    """
    return torch.cat([
        joint_pos(env, joint_ids),        # 28
        joint_vel(env, joint_ids),        # 28
        target_error(env, joint_ids),     # 28
        ball_pos_relative(env),           #  3
        ball_lin_vel(env),                #  3
        container_pos_relative(env),      #  3
        left_palm_pos_relative(env),      #  3
        right_palm_pos_relative(env),     #  3
        ball_to_left_palm(env),           #  3
        ball_to_right_palm(env),          #  3
        ball_to_container(env),           #  3
    ], dim=-1)                            # 108