import importlib
import torch
import hydra
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher

# ── Set checkpoint path here ───────────────────────────────────────────────────
CHECKPOINT_PATH = "/scratch2/danielc174/humanoid-manipulation/outputs/reorient/ppo/seed42/run_001/checkpoints/model_1000.pt"


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import manipulation.tasks

    checkpoint_path = Path(CHECKPOINT_PATH).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    eval_dir = checkpoint_path.parent.parent / "eval" / checkpoint_path.stem
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] checkpoint: {checkpoint_path}")
    print(f"[eval] output:     {eval_dir}")

    SCRIPT_KEYS = {"gym_id", "log_name", "env_cfg_module", "env_cfg_class",
                   "cameras", "viewer"}

    cameras = cfg.task.cameras
    first_cam = next(iter(cameras.values()))

    # ── Build env once ────────────────────────────────────────────────────────
    module      = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg     = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    task_dict = OmegaConf.to_container(cfg.task, resolve=True)
    for key, val in task_dict.items():
        if key in SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    env_cfg.viewer.resolution  = (cfg.video_width, cfg.video_height)
    env_cfg.viewer.env_index   = 0
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye         = tuple(first_cam.eye)
    env_cfg.viewer.lookat      = tuple(first_cam.lookat)

    env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")

    # ── Load agent ────────────────────────────────────────────────────────────
    from manipulation.algos.ppo.ppo import PPO

    obs_dim    = env.unwrapped.single_observation_space["policy"].shape[0]
    action_dim = env.unwrapped.single_action_space.shape[0]
    device     = env.unwrapped.device

    agent = PPO(obs_dim=obs_dim, action_dim=action_dim,
                num_envs=1, cfg=cfg.algo, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.network.load_state_dict(ckpt["model"])
    if "obs_mean"  in ckpt: agent.obs_mean  = ckpt["obs_mean"].to(device)
    if "obs_var"   in ckpt: agent.obs_var   = ckpt["obs_var"].to(device)
    if "obs_count" in ckpt: agent.obs_count = ckpt["obs_count"].to(device)
    agent.network.eval()

    sim = env.unwrapped.sim

    # ── Prime renderer ────────────────────────────────────────────────────────
    obs_dict, _ = env.reset(seed=0)
    for _ in range(50):
        sim.step()
    env.render()  # prime
    for _ in range(10):
        sim.step()

    # ── Episodes ──────────────────────────────────────────────────────────────
    for ep in range(cfg.num_episodes):
        obs_dict, _ = env.reset(seed=ep)
        obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)

        max_steps    = int(env.unwrapped.max_episode_length)
        total_reward = 0.0
        success      = False
        step         = 0

        # writers for each camera
        writers = {}
        for cam_name in cameras:
            path = eval_dir / f"{cam_name}_ep{ep:03d}.mp4"
            writers[cam_name] = imageio.get_writer(str(path), fps=cfg.video_fps)

        # capture first frame from all cameras
        for cam_name, cam_cfg_i in cameras.items():
            sim.set_camera_view(eye=list(cam_cfg_i.eye), target=list(cam_cfg_i.lookat))
            for _ in range(3):
                sim.step()
            frame = _get_frame(env)
            writers[cam_name].append_data(frame)

        while step < max_steps:
            with torch.no_grad():
                if cfg.deterministic:
                    action = torch.tanh(agent.network.actor(obs))
                else:
                    action, _, _ = agent.collect_step(obs)
                action = action.clamp(-1.0, 1.0)

            obs_dict, reward, terminated, timed_out, _ = env.step(action)
            obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)
            total_reward += reward[0].item()

            if terminated[0].item():
                success = True

            # capture frame from each camera
            for cam_name, cam_cfg_i in cameras.items():
                sim.set_camera_view(eye=list(cam_cfg_i.eye), target=list(cam_cfg_i.lookat))
                frame = _get_frame(env)
                writers[cam_name].append_data(frame)

            step += 1
            if (terminated | timed_out)[0].item():
                break

        for writer in writers.values():
            writer.close()

        print(f"  ep{ep:03d} | steps={step:4d} | reward={total_reward:.2f} | success={success}")

    env.close()
    simulation_app.close()


def _get_frame(env) -> np.ndarray:
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


if __name__ == "__main__":
    main()