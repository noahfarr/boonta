from ..sections import Environment
from . import environment

environment(
    Environment(
        namespace="gymnasium",
        suite="gymnasium",
        env_id="CartPole-v1",
        num_envs=16,
        kwargs=dict(num_envs="${environment.num_envs}", vectorization_mode="async"),
    ),
    name="gymnasium/cartpole",
)
