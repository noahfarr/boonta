from boonta.algorithms.bc import BCConfig

from ..sections import Optimizer
from ..store import fbuilds
from . import algorithm, offline

bc = algorithm(
    dict(
        defaults=offline,
        algorithm=fbuilds(BCConfig, batch_size=256, entropy_coefficient=0.0),
        environment=dict(num_envs=32),
        optimizer=Optimizer(lr=3e-4),
    ),
    name="bc",
)

bc.hyperparameters("brax/mujoco", environment=dict(kwargs=dict(backend="mjx")))
