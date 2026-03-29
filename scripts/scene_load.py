import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize ball_container scene spawns")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, RigidObject, AssetBaseCfg
from isaaclab.sensors import ContactSensor
from isaaclab.scene import InteractiveScene

from manipulation.tasks.ball_container.env_cfg import BallContainerSceneCfg


def main():
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1/60, gravity=(0.0, 0.0, -9.81))
    )
    sim.set_camera_view(
        eye=(2.6, -2.5, 1.8),
        target=(2.7, -0.3, 0.8),
    )

    scene_cfg = BallContainerSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    print("\n=== Scene spawned ===")
    print(f"Robot root pos:     {scene['robot'].data.root_pos_w[0]}")
    print(f"Ball root pos:      {scene['ball'].data.root_pos_w[0]}")
    print(f"Container root pos: {scene['container'].data.root_pos_w[0]}")
    print("\nSimulation running. Use GUI to inspect scene.")
    print("Close the window to exit.\n")

    while simulation_app.is_running():
        sim.step()
        scene.update(dt=1/60)

    simulation_app.close()


if __name__ == "__main__":
    main()