from ..sections import Environment
from . import environment

for env_id in (
    "ant",
    "halfcheetah",
    "hopper",
    "humanoidstandup",
    "humanoid",
    "inverted_double_pendulum",
    "inverted_pendulum",
    "pusher",
    "reacher",
    "swimmer",
    "walker2d",
):
    environment(
        Environment(namespace="brax", suite="mujoco", env_id=env_id, kwargs=dict(backend="generalized")),
        name=f"brax/mujoco/{env_id}",
    )
