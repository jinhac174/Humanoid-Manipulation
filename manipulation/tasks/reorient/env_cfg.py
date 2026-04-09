"""
Reorient task — fixed-base G1 bimanual cuboid reorientation.
Port of SAPG's Two-Arms Reorientation task (isaacgymenvs AllegroKukaTwoArmsReorientation).

Phase A: scene configuration only. Observations, rewards, and resets are stubs
that keep the env instantiable for `scripts/scene_load.py task=reorient`.
See reorient_port_plan_v2.md for the phase plan.
"""
from __future__ import annotations

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from manipulation.robots.g1 import G1_FIXED_CFG

# NOTE on USD choice: this task uses the default G1_FIXED_CFG USD
# (g1_dex3.usd via manipulation/robots/g1.py), the same one can_push uses.
# g1_dex3_head.usd is structurally incompatible with fix_root_link=True in
# IsaacLab 2.3.2 — its articulation root prim lacks RigidBodyAPI, and the
# IsaacLab fallback of finding the first rigid body link to attach the
# fixed joint to is NotImplemented. Revisit if head-mounted sensors are
# needed in Phase E; for Phase A–D the port has no head-dependent logic.


# ─────────────────────────────────────────────────────────────────────────────
# Robot body name constants (referenced from env.py and observations.py)
# ─────────────────────────────────────────────────────────────────────────────
LEFT_PALM_BODY = "left_hand_palm_link"
RIGHT_PALM_BODY = "right_hand_palm_link"

# Fingertip body names — inferred from STL filenames in assets/robots/g1/meshes.
# If the USD uses different names, these fail at robot.find_bodies() in Phase B
# with a clear error. Donor uses 4 fingertips per hand (Allegro); Dex3 has 3.
LEFT_FINGERTIP_BODIES = [
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
]
RIGHT_FINGERTIP_BODIES = [
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
]
NUM_FINGERTIPS_PER_HAND = 3  # thumb, index, middle (Dex3 has no ring)
NUM_ARMS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Scene layout constants (meters, world frame)
# ─────────────────────────────────────────────────────────────────────────────
# G1 pelvis is at z=0.76 (from G1_FIXED_CFG default). Shoulders sit roughly at
# z≈1.3; arms reach ~0.65 m from each shoulder. Table top at z=0.90 puts the
# cuboid in front of the chest, inside easy bimanual reach.
TABLE_SIZE = (1.0, 1.0, 0.78)  # x, y, z
TABLE_POS = (0.70, 0.0, 0.39)  # center; bottom at z=0, top at z=0.78
TABLE_TOP_Z = TABLE_POS[2] + TABLE_SIZE[2] / 2  # 0.78

CUBOID_SIZE = (0.05, 0.05, 0.05)  # donor's objectBaseSize = 0.05

# Cuboid spawn xy is DECOUPLED from the table center so the cube can sit
# comfortably within G1's reach envelope (~0.65 m from each shoulder). Palms
# at default pose are at x≈0.24; cube at x=0.40 puts it ~0.22 m from each
# palm, well inside reach. The cube still lands on the table because the
# table footprint is x ∈ [0.20, 1.20].
CUBOID_SPAWN_XY = (0.40, 0.0)
CUBOID_SPAWN_POS = (
    CUBOID_SPAWN_XY[0],
    CUBOID_SPAWN_XY[1],
    TABLE_TOP_Z + CUBOID_SIZE[2] / 2 + 0.05,
)
# the +0.05 above the table lets the cube drop onto the surface at episode
# start so it settles naturally instead of interpenetrating

# Target volume (donor equivalent: target_volume_origin + target_volume_extent).
# Conservative vs donor [[-0.2,0.2],[-0.5,0.5],[-0.12,0.25]] — G1 reach is
# narrower than a dual-Kuka setup. Widen in Phase E after the reach check.
TARGET_VOLUME_ORIGIN = (CUBOID_SPAWN_XY[0], CUBOID_SPAWN_XY[1], TABLE_TOP_Z + 0.15)
TARGET_VOLUME_EXTENT = (
    (-0.10, 0.10),  # x
    (-0.20, 0.20),  # y
    (-0.05, 0.15),  # z
)

# Object drop termination height (donor: object_pos_z < 0.1, i.e. "on the floor").
# Here anything below the table surface by 0.2 m counts as dropped.
OBJECT_DROP_Z = TABLE_TOP_Z - 0.20  # 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Reward scales — donor values from AllegroKuka.yaml, unchanged
# (Phase A: defined on the cfg but unused; Phase C: wired into rewards.py)
# ─────────────────────────────────────────────────────────────────────────────
LIFTING_REW_SCALE = 20.0
LIFTING_BONUS = 300.0
LIFTING_BONUS_THRESHOLD = 0.15  # m above init
KEYPOINT_REW_SCALE = 200.0
DISTANCE_DELTA_REW_SCALE = 50.0
REACH_GOAL_BONUS = 1000.0
FALL_PENALTY = 0.0

SUCCESS_TOLERANCE = 0.075
TARGET_SUCCESS_TOLERANCE = 0.01
TOLERANCE_CURRICULUM_INCREMENT = 0.9
TOLERANCE_CURRICULUM_INTERVAL = 3000
MAX_CONSECUTIVE_SUCCESSES = 50
SUCCESS_STEPS = 1
KEYPOINT_SCALE = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class ReorientSceneCfg(InteractiveSceneCfg):
    """G1 fixed-base + table + cuboid + visual goal marker."""

    # ── Robot ────────────────────────────────────────────────────────────────
    # Uses G1_FIXED_CFG exactly as defined in manipulation/robots/g1.py
    # (g1_dex3.usd). reset_robot in events.py sends G1 back to that global
    # default each episode. If the default pose turns out to be wrong for
    # bimanual reorient we see it on the first scene_load and override
    # init_state here.
    robot = G1_FIXED_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # ── Table (static rigid box) ─────────────────────────────────────────────
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.58, 0.40)
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # ── Cuboid (dynamic rigid, the manipulated object) ───────────────────────
    cuboid: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=CUBOID_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.30, 0.30)
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBOID_SPAWN_POS),
    )

    # ── Goal (visual-only cuboid at the target pose) ─────────────────────────
    # Gravity off, collisions off, non-kinematic. We write its root state in
    # events.reset_objects each reset; with no forces acting on it, it stays
    # where we put it. Matches the donor's pattern (disable_gravity=True only).
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CuboidCfg(
            size=CUBOID_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.85, 0.35),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_VOLUME_ORIGIN,
        ),
    )

    # ── Lighting ─────────────────────────────────────────────────────────────
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Env cfg
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class ReorientEnvCfg(DirectRLEnvCfg):
    scene: ReorientSceneCfg = ReorientSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 200,
        render_interval=5,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
        ),
    )

    decimation: int = 5  # 200 Hz sim / 5 = 40 Hz control
    episode_length_s: float = 10.0  # donor: 600 steps @ 60 Hz = 10 s

    # Phase B: 108-dim minimal observation (see observations.py for layout).
    observation_space = Box(low=-np.inf, high=np.inf, shape=(108,))
    state_space = 0
    action_space = Box(low=-1.0, high=1.0, shape=(28,))

    # ── Scene layout (exposed so train.py can override via YAML) ─────────────
    table_top_z: float = TABLE_TOP_Z
    cuboid_spawn_pos: tuple = CUBOID_SPAWN_POS
    target_volume_origin: tuple = TARGET_VOLUME_ORIGIN
    target_volume_extent: tuple = TARGET_VOLUME_EXTENT
    object_drop_z: float = OBJECT_DROP_Z

    # ── Donor reward scales (Phase A: unused; Phase C: wired into rewards.py) ─
    lifting_rew_scale: float = LIFTING_REW_SCALE
    lifting_bonus: float = LIFTING_BONUS
    lifting_bonus_threshold: float = LIFTING_BONUS_THRESHOLD
    keypoint_rew_scale: float = KEYPOINT_REW_SCALE
    distance_delta_rew_scale: float = DISTANCE_DELTA_REW_SCALE
    reach_goal_bonus: float = REACH_GOAL_BONUS
    fall_penalty: float = FALL_PENALTY

    # ── Donor success / curriculum (Phase A: unused; Phase D: curriculum on) ─
    success_tolerance: float = SUCCESS_TOLERANCE
    target_success_tolerance: float = TARGET_SUCCESS_TOLERANCE
    tolerance_curriculum_increment: float = TOLERANCE_CURRICULUM_INCREMENT
    tolerance_curriculum_interval: int = TOLERANCE_CURRICULUM_INTERVAL
    max_consecutive_successes: int = MAX_CONSECUTIVE_SUCCESSES
    success_steps: int = SUCCESS_STEPS
    keypoint_scale: float = KEYPOINT_SCALE
    enable_curriculum: bool = False  # Phase C: False; Phase D: True