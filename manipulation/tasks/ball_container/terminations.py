import torch
from isaaclab.envs import DirectRLEnv

# ── Container geometry — UPDATE after GUI measurement ─────────────────────────
_CONTAINER_HALF_X   = 0.15   # inner half-width  (X)
_CONTAINER_HALF_Y   = 0.20   # inner half-depth  (Y)
_CONTAINER_FLOOR_Z  = 0.00   # Z from origin to inner floor (0 if origin at floor)
_CONTAINER_RIM_Z    = 0.12   # Z from origin to top of walls


def termination_success(env: DirectRLEnv) -> torch.Tensor:
    """
    Ball is inside the open-top rectangular container.

    Checks rectangular XY bounds of inner opening and
    Z between floor and rim. Update geometry constants above
    once container USD is measured.
    """
    ball_pos      = env.ball.data.root_pos_w
    container_pos = env.container.data.root_pos_w

    dx = torch.abs(ball_pos[:, 0] - container_pos[:, 0])
    dy = torch.abs(ball_pos[:, 1] - container_pos[:, 1])
    dz = ball_pos[:, 2] - container_pos[:, 2]

    x_ok = dx < _CONTAINER_HALF_X
    y_ok = dy < _CONTAINER_HALF_Y
    z_ok = (dz > _CONTAINER_FLOOR_Z) & (dz < _CONTAINER_RIM_Z)

    return x_ok & y_ok & z_ok


def termination_ball_dropped(env: DirectRLEnv, drop_threshold: float = 0.15) -> torch.Tensor:
    """
    Ball fell more than drop_threshold below its spawn Z.
    Terminates immediately to break the death spiral of
    accumulating drop penalties for the rest of the episode.
    """
    return env.ball.data.root_pos_w[:, 2] < (env.ball_spawn_z - drop_threshold)


def termination_timeout(env: DirectRLEnv) -> torch.Tensor:
    """Episode exceeded max length."""
    return env.episode_length_buf >= env.max_episode_length