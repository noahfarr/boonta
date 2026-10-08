from boonta.algorithms.mmd import MMDConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, special
from .ippo import connectx

algorithm(
    dict(
        algorithm=fbuilds(
            MMDConfig,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.0,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            magnet_coefficient=0.01,
            magnet_decay=0.99,
            magnet_anneal=1.0,
        ),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=4096),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="mmd",
)

special("mmd", "connectx", **connectx)
