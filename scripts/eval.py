import importlib
import torch
import hydra
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from omegaconf import DictConfig

from isaaclab.app import AppLauncher

# ── Change this to evaluate a different checkpoint ────────────────────────────
CHECKPOINT_PATH = "/scratch2/danielc174/humanoid-manipulation/outputs/ball_container/ppo/seed0/run_000/checkpoints/model_500.pt"


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
    print(f"[eval] output dir: {eval_dir}")

    for cam_name, cam_cfg in cfg.scene.cameras.items():

        # ── Build env ─────────────────────────────────────────────────────
        module      = importlib.import_module(cfg.task.env_cfg_module)
        EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
        env_cfg     = EnvCfgClass()
        env_cfg.scene.num_envs = 1

        env_cfg.viewer.resolution  = (cfg.video_width, cfg.video_height)
        env_cfg.viewer.env_index   = 0
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye         = tuple(cam_cfg.eye)
        env_cfg.viewer.lookat      = tuple(cam_cfg.lookat)

        env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")

        # ── Load agent ────────────────────────────────────────────────────
        from manipulation.algos.ppo.ppo import PPO

        obs_dim    = env.unwrapped.single_observation_space["policy"].shape[0]
        action_dim = env.unwrapped.single_action_space.shape[0]
        device     = env.unwrapped.device

        agent = PPO(
            obs_dim    = obs_dim,
            action_dim = action_dim,
            num_envs   = 1,
            cfg        = cfg.algo,
            device     = device,
        )

        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.network.load_state_dict(ckpt["model"])
        if "obs_mean"  in ckpt: agent.obs_mean  = ckpt["obs_mean"].to(device)
        if "obs_var"   in ckpt: agent.obs_var   = ckpt["obs_var"].to(device)
        if "obs_count" in ckpt: agent.obs_count = ckpt["obs_count"].to(device)
        agent.network.eval()

        print(f"[eval] camera: {cam_name}")

        # ── Episodes ──────────────────────────────────────────────────────
        for ep in range(cfg.num_episodes):
            video_path = eval_dir / f"{cam_name}_ep{ep:03d}.mp4"

            obs_dict, _ = env.reset(seed=ep)
            obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)

            total_reward = 0.0
            success      = False
            step         = 0
            max_steps    = int(env.unwrapped.max_episode_length)

            with imageio.get_writer(str(video_path), fps=cfg.video_fps) as writer:
                frame = env.render()
                writer.append_data(_to_uint8(frame))

                while step < max_steps:
                    with torch.no_grad():
                        if cfg.deterministic:
                            action = torch.tanh(agent.network.actor(obs))
                        else:
                            action, _, _ = agent.collect_step(obs)
                        action = action.clamp(-1.0, 1.0)

                    obs_dict, reward, terminated, timed_out, _ = env.step(action)
                    obs = agent.normalize_obs(obs_dict["policy"], update_stats=False)

                    frame = env.render()
                    writer.append_data(_to_uint8(frame))

                    total_reward += reward[0].item()
                    done = bool((terminated | timed_out)[0].item())
                    if terminated[0].item():
                        success = True

                    step += 1
                    if done:
                        break

            print(
                f"  ep{ep:03d} | steps={step:4d} | "
                f"reward={total_reward:.2f} | success={success}"
            )

        env.close()

    simulation_app.close()


def _to_uint8(frame):
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


if __name__ == "__main__":
    main()