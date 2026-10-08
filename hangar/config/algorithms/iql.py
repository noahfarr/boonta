from boonta.algorithms.iql import IQLConfig

from ..sections import Optimizers
from ..store import fbuilds
from . import algorithm, offline, special

algorithm(
    dict(
        defaults=offline,
        algorithm=fbuilds(
            IQLConfig,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            expectile=0.7,
            beta=3.0,
            max_advantage_weight=100.0,
        ),
        environment=dict(num_envs=32),
        optimizer=Optimizers(actor_lr=3e-4, critic_lr=3e-4, value_lr=3e-4),
    ),
    name="iql",
)

special("iql", "brax/mujoco", environment=dict(kwargs=dict(backend="mjx")))
