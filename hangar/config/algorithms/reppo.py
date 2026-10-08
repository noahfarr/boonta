from boonta.algorithms.reppo import REPPOConfig

from ..sections import Optimizers, Rollout
from ..store import fbuilds
from . import algorithm

reppo = algorithm(
    dict(
        algorithm=fbuilds(
            REPPOConfig,
            num_minibatches=128,
            update_epochs=4,
            gamma=0.99,
            td_lambda=0.95,
            target_kl=0.1,
            target_entropy_scale=0.5,
            num_kl_samples=16,
            vmin=0.0,
            vmax=150.0,
            num_bins=151,
            zen_exclude=("action_dim",),
        ),
        rollout=Rollout(num_steps=128),
        environment=dict(num_envs=1024),
        optimizer=Optimizers(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, lagrangian_lr=3e-4),
    ),
    name="reppo",
)

reppo.hyperparameters(
    "brax",
    total_timesteps=50_000_000,
    rollout=dict(num_steps=32),
    environment=dict(num_envs=4096),
)
