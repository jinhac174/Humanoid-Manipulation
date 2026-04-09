"""
Insert task — fixed-base G1, HumanoidBench-inspired.
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

        # ── Asset handles (after super().__init__ so scene is ready) ─────────
        self.robot: Articulation = self.scene["robot"]
        self.peg_a: RigidObject  = self.scene["peg_a"]
        self.peg_b: RigidObject  = self.scene["peg_b"]
        self.block: RigidObject  = self.scene["block"]

        # ── Constant offset tensors ───────────────────────────────────────────
        self.block_peg_a_offset = torch.tensor(
            BLOCK_PEG_A_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)

        self.block_peg_b_offset = torch.tensor(
            BLOCK_PEG_B_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)

        self.peg_site_offset = torch.tensor(
            PEG_SITE_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)

        # ── Actuated joint IDs ────────────────────────────────────────────────
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
        self.arm_hand_joint_ids = self.actuated_joint_ids

        # ── Body IDs ──────────────────────────────────────────────────────────
        self.left_palm_body_id  = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self.right_palm_body_id = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]

        # ── Action scale ──────────────────────────────────────────────────────
        from manipulation.robots.g1 import ACTION_SCALE, ACTUATED_JOINTS
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
        self.action_scale = torch.tensor(
            [scale_map.get(name, 0.25) for name in resolved_names],
            dtype=torch.float32, device=self.device,
        )

        # ── Joint targets (accumulated delta control) ─────────────────────────
        self.joint_targets = self.robot.data.default_joint_pos[
            :, self.actuated_joint_ids
        ].clone()

        # ── Episode buffers ───────────────────────────────────────────────────
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.success_buf    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_actions   = torch.zeros(self.num_envs, len(self.actuated_joint_ids), device=self.device)

        # expose for rewards.py
        self.cfg.peg_height_target = PEG_HEIGHT_TARGET

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

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({f"reward/{k}": v.mean().item() for k, v in info.items()})

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out  = self.episode_length_buf >= self.max_episode_length - 1
        dropped   = term_fn.object_dropped(self)
        succeeded = term_fn.success(self)
        terminated = dropped | succeeded

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["done/drop_rate"]    = dropped.float().mean().item()
        self.extras["log"]["done/success_rate"] = succeeded.float().mean().item()
        self.extras["log"]["rollout/ep_reward"] = self.episode_reward.mean().item()

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_objects(self, env_ids)
        event_fn.reset_buffers(self, env_ids)
        self.joint_targets[env_ids] = self.robot.data.default_joint_pos[
            env_ids][:, self.actuated_joint_ids].clone()
        self.prev_actions[env_ids] = 0.0