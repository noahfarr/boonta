from boonta.algorithms.grpo import GRPOConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, special

algorithm(
    dict(
        algorithm=fbuilds(
            GRPOConfig,
            group_size=8,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.04,
            gamma=0.99,
        ),
        rollout=Rollout(num_steps=256),
        environment=dict(num_envs=512),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="grpo",
)

special("grpo", "gymnax/minatar", total_timesteps=50_000_000)
