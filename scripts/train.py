import re
import torch
import wandb
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from isaaclab.app import AppLauncher


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ids = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"run_(\d+)", p.name)
        if m:
            run_ids.append(int(m.group(1)))
    next_id = 0 if not run_ids else max(run_ids) + 1
    run_dir = base_dir / f"run_{next_id:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=cfg.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import manipulation.tasks

    torch.manual_seed(cfg.seed)

    # ── Run directory ──────────────────────────────────────────────────────
    base_dir = Path(cfg.log_root) / cfg.task.log_name / cfg.algo.name / f"seed{cfg.seed}"
    run_dir  = get_next_run_dir(base_dir)

    ckpt_dir  = run_dir / "checkpoints"
    hydra_dir = run_dir / "hydra"
    wandb_dir = run_dir / "wandb"
    eval_dir  = run_dir / "eval"

    for d in [ckpt_dir, hydra_dir, wandb_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[train] run dir: {run_dir}")

    # ── Save resolved config ───────────────────────────────────────────────
    OmegaConf.save(cfg, hydra_dir / "config_resolved.yaml", resolve=True)
    with open(hydra_dir / "overrides.txt", "w") as f:
        f.write("\n".join(HydraConfig.get().overrides.task))

    # ── W&B ───────────────────────────────────────────────────────────────
    run_name = f"seed{cfg.seed}_{run_dir.name}"
    wandb.init(
        project = cfg.wandb.project,
        name    = run_name,
        group   = f"{cfg.task.log_name}_{cfg.algo.name}",
        tags    = [cfg.task.log_name, cfg.algo.name, f"seed{cfg.seed}"],
        dir     = str(wandb_dir),
        mode    = cfg.wandb.mode,
        config  = OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Environment ───────────────────────────────────────────────────────
    from manipulation.tasks.ball_container.env_cfg import BallContainerEnvCfg

    env_cfg = BallContainerEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs

    env_cfg.episode_length_s   = cfg.task.episode_length_s
    env_cfg.success_radius     = cfg.task.success_radius
    env_cfg.ball_spawn_x_range = tuple(cfg.task.ball_spawn_x_range)
    env_cfg.ball_spawn_y_range = tuple(cfg.task.ball_spawn_y_range)

    env_cfg.reward_approach_weight  = cfg.task.reward_approach_weight
    env_cfg.reward_contact_weight   = cfg.task.reward_contact_weight
    env_cfg.reward_lift_weight      = cfg.task.reward_lift_weight
    env_cfg.reward_handoff_weight   = cfg.task.reward_handoff_weight
    env_cfg.reward_transport_weight = cfg.task.reward_transport_weight
    env_cfg.reward_success_weight   = cfg.task.reward_success_weight

    env_cfg.penalty_drop_weight         = cfg.task.penalty_drop_weight
    env_cfg.penalty_right_idle_weight   = cfg.task.penalty_right_idle_weight
    env_cfg.penalty_left_idle_weight    = cfg.task.penalty_left_idle_weight
    env_cfg.penalty_joint_limits_weight = cfg.task.penalty_joint_limits_weight
    env_cfg.penalty_action_rate_weight  = cfg.task.penalty_action_rate_weight
    env_cfg.penalty_joint_vel_weight    = cfg.task.penalty_joint_vel_weight

    env = gym.make(cfg.task.gym_id, cfg=env_cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    from manipulation.algos import TRAINER_REGISTRY

    trainer = TRAINER_REGISTRY[cfg.algo.name](
        env     = env,
        cfg     = cfg,
        run_dir = run_dir,
    )

    trainer.run()

    # ── Cleanup ───────────────────────────────────────────────────────────
    wandb.finish()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()