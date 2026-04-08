import gymnasium as gym

gym.register(
    id="G1-Insert-v0",
    entry_point="manipulation.tasks.insert.env:InsertEnv",
    kwargs={"cfg": None},
    disable_env_checker=True,
)