import gymnasium as gym

from manipulation.tasks.can_push.env import CanPushEnv
from manipulation.tasks.can_push.env_cfg import CanPushEnvCfg

gym.register(
    id="Can-Push",
    entry_point="manipulation.tasks.can_push.env:CanPushEnv",
    kwargs={"cfg": CanPushEnvCfg()},
)