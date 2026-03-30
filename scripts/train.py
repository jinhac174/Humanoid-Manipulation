import re
import importlib
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

    for d in ["checkpoints", "hydra", "wandb", "eval"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    print(f"[train] run dir: {run_dir}")

    # ── Save resolved config ───────────────────────────────────────────────
    OmegaConf.save(cfg, run_dir / "hydra" / "config_resolved.yaml", resolve=True)
    with open(run_dir / "hydra" / "overrides.txt", "w") as f:
        f.write("\n".join(HydraConfig.get().overrides.task))

    # ── W&B ───────────────────────────────────────────────────────────────
    run_name = f"seed{cfg.seed}_{run_dir.name}"
    wandb.init(
        project = cfg.wandb.project,
        name    = run_name,
        group   = f"{cfg.task.log_name}_{cfg.algo.name}",
        tags    = [cfg.task.log_name, cfg.algo.name, f"seed{cfg.seed}"],
        dir     = str(run_dir / "wandb"),
        mode    = cfg.wandb.mode,
        config  = OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Environment ───────────────────────────────────────────────────────
    module     = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg    = EnvCfgClass()
    env_cfg.scene.num_envs = cfg.num_envs

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