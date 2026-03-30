import re
import torch
import wandb
from pathlib import Path
from omegaconf import OmegaConf


class PPOTrainer:

    def __init__(self, env, cfg, run_dir: Path):
        self.env     = env
        self.cfg     = cfg
        self.run_dir = run_dir
        self.device  = env.unwrapped.device

        obs_dim    = env.unwrapped.single_observation_space["policy"].shape[0]
        action_dim = env.unwrapped.single_action_space.shape[0]
        num_envs   = cfg.num_envs

        from manipulation.algos.ppo.ppo import PPO
        self.agent = PPO(
            obs_dim    = obs_dim,
            action_dim = action_dim,
            num_envs   = num_envs,
            cfg        = cfg.algo,
            device     = self.device,
        )

        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, iteration: int):
        path = self.ckpt_dir / f"model_{iteration}.pt"
        torch.save({
            "model":     self.agent.network.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "obs_mean":  self.agent.obs_mean.cpu(),
            "obs_var":   self.agent.obs_var.cpu(),
            "obs_count": self.agent.obs_count.cpu(),
            "iteration": iteration,
        }, path)
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.network.load_state_dict(ckpt["model"])
        self.agent.optimizer.load_state_dict(ckpt["optimizer"])
        self.agent.obs_mean  = ckpt["obs_mean"].to(self.device)
        self.agent.obs_var   = ckpt["obs_var"].to(self.device)
        self.agent.obs_count = ckpt["obs_count"].to(self.device)
        return ckpt.get("iteration", 0)

    # ── Training loop ─────────────────────────────────────────────────────────

    def run(self):
        cfg   = self.cfg
        agent = self.agent
        env   = self.env

        obs_dict, _ = env.reset()
        obs = agent.normalize_obs(obs_dict["policy"], update_stats=True)

        max_iter = cfg.algo.max_iterations

        for iteration in range(max_iter):

            # ── Rollout ───────────────────────────────────────────────────────
            rollout_log_sums  = {}
            rollout_log_count = 0

            for _ in range(cfg.algo.num_steps_per_env):
                actions, log_probs, values = agent.collect_step(obs)

                obs_dict, rewards, terminated, timed_out, info = env.step(actions)
                next_obs = obs_dict["policy"]

                if isinstance(info, dict) and "log" in info:
                    for k, v in info["log"].items():
                        rollout_log_sums[k] = rollout_log_sums.get(k, 0.0) + float(v)
                    rollout_log_count += 1

                next_obs = agent.normalize_obs(next_obs, update_stats=True)
                dones    = (terminated | timed_out).float()

                agent.insert(obs, actions, rewards, dones, values, log_probs)
                obs = next_obs

            # ── Read metrics BEFORE update clears buffer ──────────────────────
            step_reward_mean    = agent.buffer.rewards.mean().item()
            rollout_return_mean = agent.buffer.rewards.sum(dim=0).mean().item()

            # ── Update ────────────────────────────────────────────────────────
            agent.compute_returns(obs)
            losses = agent.update()

            # ── Logging ───────────────────────────────────────────────────────
            log_info = {}
            if rollout_log_count > 0:
                log_info = {k: v / rollout_log_count for k, v in rollout_log_sums.items()}

            metrics = {
                "rollout/step_reward_mean": step_reward_mean,
                "rollout/return_mean":      rollout_return_mean,
                "train/iteration":          iteration,
                **losses,
                **log_info,
            }

            wandb.log(metrics, step=iteration)

            if iteration % 100 == 0:
                print(
                    f"[{iteration}/{max_iter}] "
                    f"ret={rollout_return_mean:.2f} "
                    f"rew={step_reward_mean:.4f} "
                    f"loss={losses['loss/total']:.4f}"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if iteration % 500 == 0 and iteration > 0:
                path = self.save_checkpoint(iteration)
                print(f"[train] checkpoint: {path}")

        # ── Final save ────────────────────────────────────────────────────────
        path = self.save_checkpoint(max_iter)
        print(f"[train] final model: {path}")