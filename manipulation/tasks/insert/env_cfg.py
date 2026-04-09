from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from manipulation.robots.g1 import G1_FIXED_CFG

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ASSET_DIR    = _PROJECT_ROOT / "assets" / "objects" / "insert"

# ── Geometry constants (from insert_normal.xml) ───────────────────────────────
# Block insertion site offsets in block-local frame (z-up, x-forward)
BLOCK_PEG_A_OFFSET = (0.0,  0.15, 0.018)   # block_peg_a site
BLOCK_PEG_B_OFFSET = (0.0, -0.15, 0.018)   # block_peg_b site
PEG_SITE_OFFSET    = (0.0,  0.0,  0.01)    # peg_a / peg_b measurement site

# ── World layout — matches MuJoCo insert_normal.xml ratios ───────────────────
# MuJoCo: robot pelvis z=0.75, table surface z=0.95  (diff = 0.20 m)
# Isaac:  robot pelvis z=0.76, table surface z=0.95  (≈ same diff)
#
# Table: 1.0 × 1.0 × 0.95 box, center at (0.60, 0, 0.475) → surface at z=0.95
# Peg bodies sit on table surface → body root at z=0.95 (geoms start at z=0.005)
# Block body center at z = 0.95 + 0.018 = 0.968 (half-height = 0.018)
TABLE_POS       = (0.60,  0.0,  0.375)
TABLE_SIZE      = (1.0,   1.0,  0.75)

PEG_A_INIT_POS  = (0.60, -0.2,  0.75)
PEG_B_INIT_POS  = (0.60,  0.2,  0.75)
BLOCK_INIT_POS  = (0.45,  0.0,  0.75)

PEG_HEIGHT_TARGET = 0.90   # 0.75 + 0.15

# Termination: object fell way below table
OBJECT_DROP_Z   = 0.50


@configclass
class InsertSceneCfg(InteractiveSceneCfg):

    robot = G1_FIXED_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=G1_FIXED_CFG.init_state.replace(
            pos=(0.03, 0.0, 0.76),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*_hip_pitch_joint":  0.0,
                ".*_hip_roll_joint":   0.0,
                ".*_hip_yaw_joint":    0.0,
                ".*_knee_joint":       0.0,
                ".*_ankle_pitch_joint":0.0,
                ".*_ankle_roll_joint": 0.0,
                "waist_.*":            0.0,
                "left_shoulder_pitch_joint":   -0.7,
                "left_shoulder_roll_joint":     0.4,
                "right_shoulder_pitch_joint":  -0.7,
                "right_shoulder_roll_joint":   -0.4,
                ".*_elbow_joint":               0.8,
            },
        ),
    )

    # ── Table (static visual + collision) ─────────────────────────────────────
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.78, 0.55)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # ── Peg A — compound USD (dark, y=-0.2) ───────────────────────────────────
    peg_a: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PegA",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_DIR / "peg_a.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PEG_A_INIT_POS),
    )

    # ── Peg B — compound USD (light, y=+0.2) ──────────────────────────────────
    peg_b: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PegB",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_DIR / "peg_b.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PEG_B_INIT_POS),
    )

    # ── Block — single collision box (matches MuJoCo collision geometry) ───────
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_DIR / "block.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BLOCK_INIT_POS),
    )

    # ── Lighting ──────────────────────────────────────────────────────────────
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class InsertEnvCfg(DirectRLEnvCfg):

    scene: InsertSceneCfg = InsertSceneCfg(
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

    decimation:        int   = 5
    episode_length_s:  float = 12.5

    observation_space = Box(low=-np.inf, high=np.inf, shape=(124,))
    action_space      = Box(low=-1.0,    high=1.0,    shape=(28,))

    # Spawn randomization
    peg_spawn_x_range:     tuple = (-0.03,  0.03)
    peg_spawn_y_range:     tuple = (-0.03,  0.03)
    block_spawn_x_range:   tuple = (-0.05,  0.05)
    block_spawn_y_range:   tuple = (-0.05,  0.05)
    block_spawn_yaw_range: tuple = (-0.3,   0.3)

    reward_scale: float = 1.0