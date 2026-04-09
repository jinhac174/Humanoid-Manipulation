"""
Reorient task — DirectRLEnv skeleton.

Phase A: scene spawns, joint targets initialized, reset puts everything back
at spawn. Observations/rewards/terminations delegate to stub functions that
return valid-shape but trivial values. Phase B will populate observations.py
with minimal measurements (palm centers, fingertip positions, object pose,
goal pose, keypoints, lifted latch, near-goal counter). Phase C wires up the
donor reward.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from manipulation.robots.g1 import ACTION_SCALE, ACTUATED_JOINTS

from .env_cfg import (
    CUBOID_SIZE,
    LEFT_FINGERTIP_BODIES,
    LEFT_PALM_BODY,
    NUM_ARMS,
    NUM_FINGERTIPS_PER_HAND,
    RIGHT_FINGERTIP_BODIES,
    RIGHT_PALM_BODY,
    ReorientEnvCfg,
)
from . import events as event_fn
from . import observations as obs_fn
from . import rewards as rew_fn
from . import terminations as term_fn


class ReorientEnv(DirectRLEnv):
    cfg: ReorientEnvCfg

    def __init__(
        self,
        cfg: ReorientEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Asset handles ────────────────────────────────────────────────────
        self.robot: Articulation = self.scene["robot"]
        self.cuboid: RigidObject = self.scene["cuboid"]
        self.goal: RigidObject = self.scene["goal"]

        # ── Actuated joint IDs (28 total: 14 arm + 14 hand, canonical order) ─
        # Order must match ACTUATED_JOINTS in manipulation/robots/g1.py:
        #   [0:7]   left arm (shoulder_pitch,roll,yaw, elbow, wrist_roll,pitch,yaw)
        #   [7:14]  right arm (same)
        #   [14:21] left hand (index_0,1, middle_0,1, thumb_0,1,2)
        #   [21:28] right hand (same)
        self.actuated_joint_ids, resolved_names = self.robot.find_joints(
            ACTUATED_JOINTS, preserve_order=True
        )
        assert len(self.actuated_joint_ids) == 28, (
            f"expected 28 actuated joints, got {len(self.actuated_joint_ids)}: "
            f"{resolved_names}"
        )

        # ── Per-joint action scale (delta control) ───────────────────────────
        self.action_scale = torch.tensor(
            ACTION_SCALE, dtype=torch.float32, device=self.device
        )

        # ── Body IDs ─────────────────────────────────────────────────────────
        self.left_palm_body_id = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self.right_palm_body_id = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]

        left_fingertip_ids = [
            self.robot.find_bodies(name)[0][0] for name in LEFT_FINGERTIP_BODIES
        ]
        right_fingertip_ids = [
            self.robot.find_bodies(name)[0][0] for name in RIGHT_FINGERTIP_BODIES
        ]
        # Shape: (num_arms, num_fingertips_per_hand) = (2, 3)
        self.fingertip_body_ids = torch.tensor(
            [left_fingertip_ids, right_fingertip_ids],
            dtype=torch.long,
            device=self.device,
        )

        # ── Target volume tensors (for goal resampling in events.py) ─────────
        self.target_volume_origin = torch.tensor(
            self.cfg.target_volume_origin, dtype=torch.float32, device=self.device
        )
        self.target_volume_min = torch.tensor(
            [e[0] for e in self.cfg.target_volume_extent],
            dtype=torch.float32,
            device=self.device,
        )
        self.target_volume_max = torch.tensor(
            [e[1] for e in self.cfg.target_volume_extent],
            dtype=torch.float32,
            device=self.device,
        )

        # ── Delta-control joint target buffer ────────────────────────────────
        self.joint_targets = self.robot.data.default_joint_pos[
            :, self.actuated_joint_ids
        ].clone()

        # ── Task state buffers (Phase B/C will populate these) ───────────────
        # Keeping the allocations here so they exist from t=0; Phase B fills
        # them in compute_observations, Phase C uses them in rewards.
        self.lifted_object = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.near_goal_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.closest_fingertip_dist = -torch.ones(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            dtype=torch.float32,
            device=self.device,
        )
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # ── Phase C: goal-reset queue and per-step near-goal flag ────────────
        # reset_goal_buf is set in _get_dones when an env hits a success and
        # consumed in the NEXT _pre_physics_step (donor's pattern: goal pose
        # write happens before the next physics step so the new goal is in
        # buffers by the time _get_dones runs again). near_goal is computed in
        # _get_dones and read by rewards.compute_reward to fire the success
        # bonus on the same step.
        self.reset_goal_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.near_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # ── Phase B: per-step task state (updated in observations.compute_task_state) ─
        self.object_init_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.palm_center_pos = torch.zeros(
            self.num_envs, NUM_ARMS, 3, dtype=torch.float32, device=self.device
        )
        self.fingertip_pos = torch.zeros(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self.curr_fingertip_distances = torch.zeros(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            dtype=torch.float32,
            device=self.device,
        )
        self.obj_keypoint_pos = torch.zeros(
            self.num_envs, 4, 3, dtype=torch.float32, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            self.num_envs, 4, 3, dtype=torch.float32, device=self.device
        )
        self.keypoints_max_dist = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # ── Constant: object-local keypoint offsets (4 diagonal corners) ─────
        # Donor uses [±1, ±1, ±1] on 4 of the 8 corners, scaled by
        # object_base_size * keypoint_scale / 2. For our 5 cm cube with
        # keypoint_scale=1.5 this places each keypoint at ±0.0375 m per axis.
        _kp_corners = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoint_offsets = _kp_corners * (
            CUBOID_SIZE[0] * self.cfg.keypoint_scale / 2.0
        )  # (4, 3)

        # ── Phase A only: print a reach-envelope sanity check once ───────────
        self._print_reach_envelope_once()

    # ─────────────────────────────────────────────────────────────────────────
    # Step pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Consume any goal resets queued by the previous step's _get_dones.
        # The new goal pose is written to sim BEFORE the decimation loop runs,
        # so it's in physx buffers in time for this step's _get_dones.
        goal_reset_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if goal_reset_ids.numel() > 0:
            event_fn.reset_goal_only(self, goal_reset_ids)
            self.reset_goal_buf[goal_reset_ids] = False

        self.actions = actions.clamp(-1.0, 1.0)
        self.joint_targets = self.joint_targets + self.actions * self.action_scale
        lo = self.robot.data.soft_joint_pos_limits[:, self.actuated_joint_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self.actuated_joint_ids, 1]
        self.joint_targets = self.joint_targets.clamp(lo, hi)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.joint_targets,
            joint_ids=self.actuated_joint_ids,
        )

    def _get_observations(self) -> dict:
        # Second compute_task_state call. _reset_idx + sim.forward() runs
        # between _get_rewards and _get_observations, so envs that reset this
        # step now have fresh body positions. Recompute their FK/keypoints so
        # the obs tensor sees the post-reset state, not the pre-reset state
        # cached during _get_dones.
        obs_fn.compute_task_state(self)
        return {"policy": obs_fn.get_observations(self)}

    def _get_rewards(self) -> torch.Tensor:
        return rew_fn.compute_reward(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Refresh task state (palm/fingertip FK, keypoints, etc.) from
        #    current sim. Must run BEFORE rewards and observations read it.
        obs_fn.compute_task_state(self)

        # 2. Success detection. Donor formula:
        #       near_goal = keypoints_max_dist <= success_tolerance * keypoint_scale
        #       near_goal_steps += near_goal
        #       is_success = near_goal_steps >= success_steps
        keypoint_success_tol = self.cfg.success_tolerance * self.cfg.keypoint_scale
        self.near_goal = self.keypoints_max_dist <= keypoint_success_tol
        self.near_goal_steps = self.near_goal_steps + self.near_goal.int()
        is_success = self.near_goal_steps >= self.cfg.success_steps

        # 3. Successes counter (per-episode), and queue goal-only reset for
        #    successful envs to be consumed at the start of the next step.
        self.successes = self.successes + is_success.float()
        self.reset_goal_buf = self.reset_goal_buf | is_success

        # 4. Clear progress trackers for envs that just succeeded so the next
        #    step doesn't double-fire is_success on the same goal pose. The
        #    actual new-goal pose write happens in next step's _pre_physics_step.
        self.near_goal_steps = torch.where(
            is_success, torch.zeros_like(self.near_goal_steps), self.near_goal_steps
        )
        self.closest_keypoint_max_dist = torch.where(
            is_success,
            -torch.ones_like(self.closest_keypoint_max_dist),
            self.closest_keypoint_max_dist,
        )

        # 5. Terminations (drop, max-consecutive-successes, timeout).
        return term_fn.compute_dones(self)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_objects(self, env_ids)
        event_fn.reset_buffers(self, env_ids)
        self.joint_targets[env_ids] = self.robot.data.default_joint_pos[
            env_ids
        ][:, self.actuated_joint_ids].clone()

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A diagnostics
    # ─────────────────────────────────────────────────────────────────────────
    def _print_reach_envelope_once(self) -> None:
        """
        Print palm positions and target volume corners so we can visually
        verify that both hands can reach the target volume. Runs once at init.
        """
        # After super().__init__, the scene has been stepped once, so body
        # positions are populated.
        left_palm_w = self.robot.data.body_pos_w[0, self.left_palm_body_id, :]
        right_palm_w = self.robot.data.body_pos_w[0, self.right_palm_body_id, :]
        robot_root_w = self.robot.data.root_pos_w[0, :]

        origin = self.target_volume_origin
        extent_min = self.target_volume_min
        extent_max = self.target_volume_max
        tv_min = origin + extent_min
        tv_max = origin + extent_max

        # 8 corners of the target volume
        corners = torch.tensor(
            [
                [tv_min[0], tv_min[1], tv_min[2]],
                [tv_min[0], tv_min[1], tv_max[2]],
                [tv_min[0], tv_max[1], tv_min[2]],
                [tv_min[0], tv_max[1], tv_max[2]],
                [tv_max[0], tv_min[1], tv_min[2]],
                [tv_max[0], tv_min[1], tv_max[2]],
                [tv_max[0], tv_max[1], tv_min[2]],
                [tv_max[0], tv_max[1], tv_max[2]],
            ],
            device=self.device,
        )

        left_dists = torch.norm(corners - left_palm_w.unsqueeze(0), dim=-1)
        right_dists = torch.norm(corners - right_palm_w.unsqueeze(0), dim=-1)

        print("=" * 72)
        print("REORIENT PHASE A — REACH ENVELOPE CHECK (env 0)")
        print("=" * 72)
        print(f"  robot root pos:   {robot_root_w.tolist()}")
        print(f"  left  palm pos:   {left_palm_w.tolist()}")
        print(f"  right palm pos:   {right_palm_w.tolist()}")
        print(f"  table top z:      {self.cfg.table_top_z:.3f}")
        print(f"  cuboid spawn:     {self.cfg.cuboid_spawn_pos}")
        print(f"  target vol min:   {tv_min.tolist()}")
        print(f"  target vol max:   {tv_max.tolist()}")
        print(
            f"  left  palm → TV corners:  "
            f"min={left_dists.min().item():.3f}  max={left_dists.max().item():.3f}"
        )
        print(
            f"  right palm → TV corners:  "
            f"min={right_dists.min().item():.3f}  max={right_dists.max().item():.3f}"
        )
        print("  (G1 arm reach is ~0.65 m from shoulder; both palms should")
        print("   have max distance < ~0.70 m to the furthest corner)")
        print("=" * 72)