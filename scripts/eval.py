import importlib
import torch
import hydra
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher

# ── Set checkpoint path here ───────────────────────────────────────────────────
CHECKPOINT_PATH = "/scratch2/danielc174/humanoid-manipulation/outputs/can_push/ppo/seed0/run_000/checkpoints/model_final.pt"


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=cfg.headless, enable_cameras=True)
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

    for cam_name, cam_cfg in cfg.task.cameras.items():

        module      = importlib.import_module(cfg.task.env_cfg_module)
        EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
        env_cfg     = EnvCfgClass()
        env_cfg.scene.num_envs = 1

        task_dict = OmegaConf.to_container(cfg.task, resolve=True)
        SCRIPT_KEYS = {"gym_id", "log_name", "env_cfg_module", "env_cfg_class", "cameras", "viewer"}
        for key, val in task_dict.items():
            if key in SCRIPT_KEYS:
                continue
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, val)

        env_cfg.viewer.resolution  = (cfg.video_width, cfg.video_height)
        env_cfg.viewer.env_index   = 0
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye         = tuple(cam_cfg.eye)
        env_cfg.viewer.lookat      = tuple(cam_cfg.lookat)

        env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")

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

        print(f"[eval] camera: {cam_name}")

        for ep in range(cfg.num_episodes):
            video_path = eval_dir / f"{cam_name}_ep{ep:03d}.mp4"
            obs_dict, _ = env.reset(seed=ep)
            obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)

            total_reward = 0.0
            success      = False
            step         = 0
            max_steps    = int(env.unwrapped.max_episode_length)

            with imageio.get_writer(str(video_path), fps=cfg.video_fps) as writer:
                writer.append_data(_to_uint8(env.render()))

                while step < max_steps:
                    with torch.no_grad():
                        if cfg.deterministic:
                            action = torch.tanh(agent.network.actor(obs))
                        else:
                            action, _, _ = agent.collect_step(obs)
                        action = action.clamp(-1.0, 1.0)

                    obs_dict, reward, terminated, timed_out, _ = env.step(action)
                    obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)
                    writer.append_data(_to_uint8(env.render()))

                    total_reward += reward[0].item()
                    if terminated[0].item():
                        success = True
                    step += 1
                    if (terminated | timed_out)[0].item():
                        break

            print(f"  ep{ep:03d} | steps={step:4d} | reward={total_reward:.2f} | success={success}")

        env.close()

    simulation_app.close()


def _to_uint8(frame):
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