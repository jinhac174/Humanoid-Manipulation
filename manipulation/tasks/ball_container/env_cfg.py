from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from manipulation.robots.g1 import G1_FIXED_CFG

# ── Asset paths ───────────────────────────────────────────────────────────────
_ASSET_ROOT  = Path(__file__).resolve().parents[3] / "assets"
_SCENES_DIR  = _ASSET_ROOT / "scenes"
_OBJECTS_DIR = _ASSET_ROOT / "objects"


@configclass
class BallContainerSceneCfg(InteractiveSceneCfg):

    robot = G1_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    left_hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    right_hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    kitchen: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_SCENES_DIR / "kitchen.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECTS_DIR / "ball.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.40406, -0.24417, 0.77921)),
    )

    container: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Container",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECTS_DIR / "container.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.05537, -0.26535, 0.75)),
    )


@configclass
class BallContainerEnvCfg(DirectRLEnvCfg):

    scene: BallContainerSceneCfg = BallContainerSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    decimation: int = 2
    episode_length_s: float = 20.0
    success_radius: float = 0.10

    observation_space = Box(low=-np.inf, high=np.inf, shape=(108,))
    action_space      = Box(low=-1.0,    high=1.0,    shape=(28,))