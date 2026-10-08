from ..sections import Environment
from . import environment

environment(
    Environment(namespace="xland_minigrid", suite="minigrid", env_id="MiniGrid-Empty-5x5"),
    name="xland_minigrid/minigrid/empty_5x5",
)
environment(
    Environment(
        namespace="xland_minigrid", suite="xland", env_id="XLand-MiniGrid-R1-9x9", kwargs=dict(benchmark="trivial-21k")
    ),
    name="xland_minigrid/xland/r1_9x9",
)
