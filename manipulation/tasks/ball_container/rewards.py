import torch
from isaaclab.envs import DirectRLEnv

# ── Container geometry — UPDATE after GUI measurement ─────────────────────────
_CONTAINER_HALF_X   = 0.15
_CONTAINER_HALF_Y   = 0.20
_CONTAINER_FLOOR_Z  = 0.00
_CONTAINER_RIM_Z    = 0.12


# ── Phase indicators ──────────────────────────────────────────────────────────

def ball_lifted(env: DirectRLEnv, min_height: float = 0.05) -> torch.Tensor:
    """1.0 if ball is lifted more than min_height above its spawn Z."""
    height = env.ball.data.root_pos_w[:, 2] - env.ball_spawn_z
    return (height > min_height).float()


def handoff_ready(env: DirectRLEnv, threshold: float = 0.10) -> torch.Tensor:
    """1.0 if right palm is within threshold of ball."""
    right_palm = env.robot.data.body_pos_w[:, env.right_palm_idx, :]
    dist = torch.norm(right_palm - env.ball.data.root_pos_w, dim=-1)
    return (dist < threshold).float()


def ball_in_container(env: DirectRLEnv) -> torch.Tensor:
    """1.0 if ball is inside the open-top rectangular container."""
    ball_pos      = env.ball.data.root_pos_w
    container_pos = env.container.data.root_pos_w

    dx = torch.abs(ball_pos[:, 0] - container_pos[:, 0])
    dy = torch.abs(ball_pos[:, 1] - container_pos[:, 1])
    dz = ball_pos[:, 2] - container_pos[:, 2]

    x_ok = dx < _CONTAINER_HALF_X
    y_ok = dy < _CONTAINER_HALF_Y
    z_ok = (dz > _CONTAINER_FLOOR_Z) & (dz < _CONTAINER_RIM_Z)

    return (x_ok & y_ok & z_ok).float()


# ── Reward terms ──────────────────────────────────────────────────────────────

def reward_approach(env: DirectRLEnv) -> torch.Tensor:
    """
    Left palm approaches ball. Always active — no phase gate.
    """
    left_palm = env.robot.data.body_pos_w[:, env.left_palm_idx, :]
    ball_pos  = env.ball.data.root_pos_w

    xy_dist   = torch.norm(left_palm[:, :2] - ball_pos[:, :2], dim=-1)
    z_offset  = left_palm[:, 2] - ball_pos[:, 2]
    full_dist = torch.norm(left_palm - ball_pos, dim=-1)

    reach_progress = torch.clamp(
        env.prev_left_dist - full_dist, min=0.0, max=0.02
    ) / 0.02

    return (
        0.45 * torch.exp(-8.0  * xy_dist)
        + 0.20 * torch.exp(-18.0 * torch.abs(z_offset - 0.04))
        + 0.35 * reach_progress
    )


def reward_lift(env: DirectRLEnv) -> torch.Tensor:
    """Left hand lifts ball above spawn height."""
    height = env.ball.data.root_pos_w[:, 2] - env.ball_spawn_z

    lift_progress = torch.clamp(
        height - env.prev_ball_height, min=0.0, max=0.02
    ) / 0.02

    return (
        0.55 * torch.clamp(height / 0.10, 0.0, 1.0)
        + 0.45 * lift_progress
    )


def reward_handoff(env: DirectRLEnv) -> torch.Tensor:
    """Right palm approaches ball after lift. Gated on ball_lifted."""
    lifted = ball_lifted(env)

    right_palm = env.robot.data.body_pos_w[:, env.right_palm_idx, :]
    left_palm  = env.robot.data.body_pos_w[:, env.left_palm_idx, :]
    ball_pos   = env.ball.data.root_pos_w

    right_dist = torch.norm(right_palm - ball_pos, dim=-1)
    left_dist  = torch.norm(left_palm  - ball_pos, dim=-1)

    return lifted * (
        0.70 * torch.exp(-5.0 * right_dist)
        + 0.30 * torch.exp(-5.0 * left_dist)
    )


def reward_transport(env: DirectRLEnv) -> torch.Tensor:
    """Ball moves toward container after lift and handoff."""
    lifted = ball_lifted(env)
    ready  = handoff_ready(env)

    ball_pos      = env.ball.data.root_pos_w
    container_pos = env.container.data.root_pos_w

    xy_dist = torch.norm(
        ball_pos[:, :2] - container_pos[:, :2], dim=-1
    )

    transport_progress = torch.clamp(
        env.prev_ball_to_container - xy_dist, min=0.0, max=0.02
    ) / 0.02

    return lifted * ready * (
        0.60 * torch.exp(-4.0 * xy_dist)
        + 0.40 * transport_progress
    )


def reward_success(env: DirectRLEnv) -> torch.Tensor:
    """Ball is inside the container."""
    return ball_in_container(env)


# ── Penalty terms ─────────────────────────────────────────────────────────────

def penalty_drop(env: DirectRLEnv, drop_threshold: float = 0.08) -> torch.Tensor:
    """Ball fell below spawn height by more than drop_threshold."""
    dropped = env.ball.data.root_pos_w[:, 2] < (env.ball_spawn_z - drop_threshold)
    return dropped.float()


def penalty_right_idle(env: DirectRLEnv) -> torch.Tensor:
    """Right arm deviates from nominal pre-lift. Includes velocity damping."""
    pre_lift = 1.0 - ball_lifted(env)

    joint_pos = env.robot.data.joint_pos[:, env.actuated_joint_ids]
    joint_vel = env.robot.data.joint_vel[:, env.actuated_joint_ids]

    right_pos     = joint_pos[:, env.right_arm_slice]
    right_nominal = env.nominal_joint_pos[:, env.right_arm_slice]
    right_vel     = joint_vel[:, env.right_arm_slice]

    deviation = (right_pos - right_nominal).pow(2).mean(dim=-1)
    velocity  = right_vel.pow(2).mean(dim=-1)

    return pre_lift * (deviation + 0.3 * velocity)


def penalty_left_idle(env: DirectRLEnv) -> torch.Tensor:
    """Left arm deviates from nominal post-lift and handoff."""
    lifted = ball_lifted(env)
    ready  = handoff_ready(env)

    joint_pos    = env.robot.data.joint_pos[:, env.actuated_joint_ids]
    left_pos     = joint_pos[:, env.left_arm_slice]
    left_nominal = env.nominal_joint_pos[:, env.left_arm_slice]

    deviation = (left_pos - left_nominal).pow(2).mean(dim=-1)

    return lifted * ready * deviation


def penalty_joint_limits(env: DirectRLEnv) -> torch.Tensor:
    """Soft margin violation near joint limits."""
    joint_pos = env.robot.data.joint_pos[:, env.actuated_joint_ids]

    joint_range = (env.actuated_joint_upper - env.actuated_joint_lower).clamp(min=1e-6)
    margin = 0.10 * joint_range

    lower_viol = torch.relu((env.actuated_joint_lower + margin) - joint_pos) / margin
    upper_viol = torch.relu(joint_pos - (env.actuated_joint_upper - margin)) / margin

    return (lower_viol + upper_viol).mean(dim=-1)


def penalty_action_rate(env: DirectRLEnv) -> torch.Tensor:
    """Penalize jerk — change in action between steps."""
    return (env.actions - env.prev_actions).pow(2).mean(dim=-1)


def penalty_joint_vel(env: DirectRLEnv) -> torch.Tensor:
    """Penalize fast joint motion."""
    joint_vel = env.robot.data.joint_vel[:, env.actuated_joint_ids]
    return joint_vel.pow(2).mean(dim=-1)