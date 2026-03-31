import torch
from isaaclab.envs import DirectRLEnv


def termination_success(env: DirectRLEnv) -> torch.Tensor:
    """Can center is within target radius (XY only)."""
    xy_dist = torch.norm(
        env.can.data.root_pos_w[:, :2] - env.target_pos_w[:, :2], dim=-1
    )
    return xy_dist < env.cfg.success_radius


def termination_can_dropped(env: DirectRLEnv, drop_threshold: float = 0.15) -> torch.Tensor:
    """Can fell more than drop_threshold below its spawn Z."""
    return env.can.data.root_pos_w[:, 2] < (env.can_spawn_z - drop_threshold)


def termination_timeout(env: DirectRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length