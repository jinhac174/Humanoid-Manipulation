"""
Reset events for the reorient task.

Phase A handles:
    - reset_robot:   default joint pos (from G1_FIXED_CFG), zero velocity
    - reset_objects: cuboid → spawn pose with small xy noise + random yaw
                     goal   → random pose inside target_volume, random rotation
    - reset_buffers: clear lifted_object, near_goal_steps, successes,
                     closest_fingertip_dist, closest_keypoint_max_dist

Phase B/C may add extra reset logic (e.g. clearing episodic reward accumulators)
but the shape established here is stable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

if TYPE_CHECKING:
    from .env import ReorientEnv


def _random_quat_wxyz(num: int, device: torch.device) -> torch.Tensor:
    """
    Uniformly random unit quaternion in (w, x, y, z) convention.

    Ported from allegro_kuka_two_arms.py::get_random_quat (Marsaglia's method).
    Avoids depending on an IsaacLab utility name that varies across 2.x minor
    versions.
    """
    uvw = torch.rand(num, 3, device=device)
    two_pi = 2.0 * math.pi
    sqrt_1_mu = torch.sqrt(1.0 - uvw[:, 0])
    sqrt_mu = torch.sqrt(uvw[:, 0])
    q_w = sqrt_1_mu * torch.sin(two_pi * uvw[:, 1])
    q_x = sqrt_1_mu * torch.cos(two_pi * uvw[:, 1])
    q_y = sqrt_mu * torch.sin(two_pi * uvw[:, 2])
    q_z = sqrt_mu * torch.cos(two_pi * uvw[:, 2])
    # Stack as (w, x, y, z) — IsaacLab convention
    return torch.stack([q_w, q_x, q_y, q_z], dim=-1)


def reset_robot(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """Reset all joints to their default values, zero velocities."""
    default_joint_pos = env.robot.data.default_joint_pos[env_ids]
    default_joint_vel = env.robot.data.default_joint_vel[env_ids]
    env.robot.write_joint_state_to_sim(
        default_joint_pos, default_joint_vel, env_ids=env_ids
    )

    # Fixed-base: root never moves, but we still write it to sync internal buffers.
    default_root_state = env.robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    env.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)


def _sample_goal_pose(
    env: "ReorientEnv", env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a uniform random pose inside the target volume for each env in
    `env_ids`. Returns (goal_pos_w, goal_quat_w) of shape (num, 3) and (num, 4).

    Used by both reset_objects (full episode reset) and reset_goal_only
    (success-triggered mid-episode goal swap).
    """
    num = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    rand01 = torch.rand(num, 3, device=env.device)
    tv_min = env.target_volume_min.unsqueeze(0)
    tv_max = env.target_volume_max.unsqueeze(0)
    goal_rel = tv_min + rand01 * (tv_max - tv_min)
    goal_pos_w = env.target_volume_origin.unsqueeze(0) + goal_rel + env_origins
    goal_quat_w = _random_quat_wxyz(num, device=env.device)
    return goal_pos_w, goal_quat_w


def reset_objects(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """
    Reset the cuboid to its spawn pose (with small noise) and the goal to
    a random pose inside the target volume.
    """
    num_reset = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    # ── Cuboid ───────────────────────────────────────────────────────────────
    cuboid_spawn = torch.tensor(
        env.cfg.cuboid_spawn_pos, dtype=torch.float32, device=env.device
    )
    # Small xy position noise (±1 cm) so successive episodes aren't identical.
    pos_noise = torch.zeros(num_reset, 3, device=env.device)
    pos_noise[:, :2] = (torch.rand(num_reset, 2, device=env.device) - 0.5) * 0.02
    cuboid_pos_w = cuboid_spawn.unsqueeze(0) + pos_noise + env_origins

    # Random yaw around world-z so the cuboid doesn't always start axis-aligned.
    # (Phase A: small rotation only; Phase C will use full random_orientation.)
    yaw = (torch.rand(num_reset, device=env.device) - 0.5) * 0.6  # ±0.3 rad
    half = 0.5 * yaw
    cuboid_quat_w = torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
        dim=-1,
    )  # (w, x, y, z)

    cuboid_state = torch.zeros(num_reset, 13, device=env.device)
    cuboid_state[:, 0:3] = cuboid_pos_w
    cuboid_state[:, 3:7] = cuboid_quat_w
    # linear + angular velocities stay zero
    env.cuboid.write_root_pose_to_sim(cuboid_state[:, 0:7], env_ids=env_ids)
    env.cuboid.write_root_velocity_to_sim(cuboid_state[:, 7:13], env_ids=env_ids)

    # Record the cuboid's world-frame initial position so the lift latch in
    # rewards._lifting_reward can reference it: z_lift = object_z - object_init_z.
    env.object_init_pos_w[env_ids] = cuboid_pos_w

    # ── Goal ─────────────────────────────────────────────────────────────────
    goal_pos_w, goal_quat_w = _sample_goal_pose(env, env_ids)
    goal_state = torch.zeros(num_reset, 13, device=env.device)
    goal_state[:, 0:3] = goal_pos_w
    goal_state[:, 3:7] = goal_quat_w
    env.goal.write_root_pose_to_sim(goal_state[:, 0:7], env_ids=env_ids)
    env.goal.write_root_velocity_to_sim(goal_state[:, 7:13], env_ids=env_ids)


def reset_goal_only(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """
    Resample only the goal pose for the given envs (NOT the cuboid, NOT the
    robot). Called from _pre_physics_step when an env has hit a success on
    the previous step. Donor equivalent: reset_target_pose.

    Note: near_goal_steps and closest_keypoint_max_dist for these envs were
    already cleared in _get_dones at the moment of success detection, so this
    function only writes the new goal pose to sim.
    """
    num = env_ids.numel()
    goal_pos_w, goal_quat_w = _sample_goal_pose(env, env_ids)

    goal_state = torch.zeros(num, 13, device=env.device)
    goal_state[:, 0:3] = goal_pos_w
    goal_state[:, 3:7] = goal_quat_w
    env.goal.write_root_pose_to_sim(goal_state[:, 0:7], env_ids=env_ids)
    env.goal.write_root_velocity_to_sim(goal_state[:, 7:13], env_ids=env_ids)


def reset_buffers(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """Clear all task-specific state tensors for the given envs."""
    env.lifted_object[env_ids] = False
    env.near_goal_steps[env_ids] = 0
    env.successes[env_ids] = 0.0
    env.reset_goal_buf[env_ids] = False
    env.near_goal[env_ids] = False
    # Sentinel -1 triggers lazy init on first obs computation (see donor code).
    env.closest_fingertip_dist[env_ids] = -1.0
    env.closest_keypoint_max_dist[env_ids] = -1.0