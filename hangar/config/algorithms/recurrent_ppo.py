from boonta.algorithms.recurrent_ppo import RecurrentPPOConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, recurrent

recurrent_ppo = algorithm(
    dict(
        defaults=recurrent,
        algorithm=fbuilds(
            RecurrentPPOConfig,
            num_minibatches=8,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantage=False,
        ),
        rollout=Rollout(num_steps=128),
        environment=dict(num_envs=64),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="recurrent_ppo",
)

recurrent_ppo.hyperparameters(
    "gymnax/minatar",
    total_timesteps=10_000_000,
    environment=dict(num_envs=512),
    rollout=dict(num_steps=16),
)
recurrent_ppo.hyperparameters(
    "peanut_gb/pokemon_red",
    total_timesteps=1_000_000_000,
    training=dict(num_epochs=635),
    environment=dict(num_envs=1024),
    rollout=dict(num_steps=128),
    cell=dict(features=256, dtype="bfloat16"),
    network=dict(dtype="bfloat16"),
    algorithm=dict(
        num_minibatches=16,
        update_epochs=1,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=2.5e-4,
        value_coefficient=0.5,
        gamma=0.999,
        gae_lambda=0.95,
        normalize_advantage=False,
    ),
    optimizer=dict(lr=5e-3, anneal=True, min_lr_ratio=0.0, beta=0.95, weight_decay=0.01, max_grad_norm=0.5),
)
recurrent_ppo.hyperparameters(
    "wordle",
    environment=dict(num_envs=32),
    algorithm=dict(num_minibatches=16),
    network=dict(repo_id="Qwen/Qwen3-0.6B-Base", dtype="bfloat16", param_dtype="bfloat16"),
)
