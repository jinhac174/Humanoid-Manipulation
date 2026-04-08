"""
Insert task — fixed-base G1, HumanoidBench-inspired.

Action space (28-dim): delta joint positions for arm+hand joints.
    [0:7]   left  arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    [7:14]  right arm
    [14:21] left  hand (index_0/1, middle_0/1, thumb_0/1/2)
    [21:28] right hand

Observation space (124-dim): see observations.py

Reward: HumanoidBench insert reward without standing terms.
"""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .env_cfg import (
    InsertEnvCfg,
    BLOCK_PEG_A_OFFSET,
    BLOCK_PEG_B_OFFSET,
    PEG_SITE_OFFSET,
    PEG_HEIGHT_TARGET,
)
from . import observations as obs_fn
from . import rewards as rew_fn
from . import events as event_fn
from . import terminations as term_fn

LEFT_PALM_BODY  = "left_hand_palm_link"
RIGHT_PALM_BODY = "right_hand_palm_link"


class InsertEnv(DirectRLEnv):

    cfg: InsertEnvCfg

    def __init__(self, cfg: InsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Constant offset tensors ───────────────────────────────────────────
        self.block_peg_a_offset = torch.tensor(
            BLOCK_PEG_A_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (N,3)

        self.block_peg_b_offset = torch.tensor(
            BLOCK_PEG_B_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (N,3)

        self.peg_site_offset = torch.tensor(
            PEG_SITE_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (N,3)

        # ── Asset handles ─────────────────────────────────────────────────────
        self.robot: Articulation = self.scene["robot"]
        self.peg_a: RigidObject  = self.scene["peg_a"]
        self.peg_b: RigidObject  = self.scene["peg_b"]
        self.block: RigidObject  = self.scene["block"]

        # ── Actuated joint IDs (28: 7 left arm + 7 right arm + 14 hands) ─────
        self.actuated_joint_ids, _ = self.robot.find_joints([
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
            ".*_hand_index_0_joint",
            ".*_hand_index_1_joint",
            ".*_hand_middle_0_joint",
            ".*_hand_middle_1_joint",
            ".*_hand_thumb_0_joint",
            ".*_hand_thumb_1_joint",
            ".*_hand_thumb_2_joint",
        ])
        # same set used for small_control torque penalty
        self.arm_hand_joint_ids = self.actuated_joint_ids

        # ── Body IDs ──────────────────────────────────────────────────────────
        self.left_palm_body_id  = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self.right_palm_body_id = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]

        # ── Action scale per joint ────────────────────────────────────────────
        from manipulation.robots.g1 import ACTION_SCALE, ACTUATED_JOINTS
        # find position of each actuated joint in our joint_ids list
        self.action_scale = torch.zeros(len(self.actuated_joint_ids), device=self.device)
        _, resolved_names = self.robot.find_joints([
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",   ".*_elbow_joint",
            ".*_wrist_roll_joint",     ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
            ".*_hand_index_0_joint",   ".*_hand_index_1_joint",
            ".*_hand_middle_0_joint",  ".*_hand_middle_1_joint",
            ".*_hand_thumb_0_joint",   ".*_hand_thumb_1_joint",
            ".*_hand_thumb_2_joint",
        ])
        scale_map = {name: scale for name, scale in zip(ACTUATED_JOINTS, ACTION_SCALE)}
        for i, name in enumerate(resolved_names):
            self.action_scale[i] = scale_map.get(name, 0.25)

        # ── Accumulated joint position targets ────────────────────────────────
        self.joint_targets = self.robot.data.default_joint_pos[
            :, self.actuated_joint_ids
        ].clone()

        # ── Episode buffers ───────────────────────────────────────────────────
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.success_buf    = torch.zeros(self.num_envs, dtype=torch.bool,
                                          device=self.device)

        self.prev_actions = torch.zeros(self.num_envs, 28, device=self.device)

        # expose peg_height_target on cfg for rewards.py
        self.cfg.peg_height_target = PEG_HEIGHT_TARGET

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.scene.articulations["robot"] = self.scene["robot"]
        self.scene.rigid_objects["peg_a"] = self.scene["peg_a"]
        self.scene.rigid_objects["peg_b"] = self.scene["peg_b"]
        self.scene.rigid_objects["block"] = self.scene["block"]
        super()._setup_scene()

    # ── Step pipeline ─────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
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
        return {"policy": obs_fn.get_observations(self)}

    def _get_rewards(self) -> torch.Tensor:
        reward, info = rew_fn.compute_reward(self)
        self.episode_reward += reward
        self.extras.update({f"reward/{k}": v.mean() for k, v in info.items()})
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out  = self.episode_length_buf >= self.max_episode_length - 1
        dropped   = term_fn.object_dropped(self)
        succeeded = term_fn.success(self)
        terminated = dropped | succeeded

        self.extras["done/drop_rate"]    = dropped.float().mean()
        self.extras["done/success_rate"] = succeeded.float().mean()
        self.extras["rollout/ep_reward"] = self.episode_reward.mean()

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_objects(self, env_ids)
        event_fn.reset_buffers(self, env_ids)
        # sync accumulated targets with freshly reset joint positions
        self.joint_targets[env_ids] = self.robot.data.default_joint_pos[
            env_ids][:, self.actuated_joint_ids].clone()