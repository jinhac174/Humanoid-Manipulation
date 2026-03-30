import gymnasium as gym

from manipulation.tasks.ball_container.env import BallContainerEnv
from manipulation.tasks.ball_container.env_cfg import BallContainerEnvCfg

gym.register(
    id="Ball-Container",
    entry_point="manipulation.tasks.ball_container.env:BallContainerEnv",
    kwargs={"cfg": BallContainerEnvCfg()},
)