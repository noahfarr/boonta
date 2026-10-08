from ..sections import Environment
from . import environment

environment(
    Environment(
        namespace="connectx",
        suite="connectx",
        env_id="connectx",
        num_envs=1024,
        kwargs=dict(rows=6, columns=7, inarow=4),
    ),
    name="connectx/connectx",
)
