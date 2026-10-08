from boonta.algorithms.ppo import PPOConfig
from boonta.artisans import Gauntlet
from boonta.environments.connectx import Negamax, uniform

from ..sections import Optimizer, Rollout
from ..store import builds, fbuilds
from . import algorithm, search, special

ippo = dict(
    algorithm=fbuilds(
        PPOConfig,
        num_minibatches=16,
        update_epochs=2,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        gamma=0.99,
        gae_lambda=0.95,
    ),
    rollout=Rollout(num_steps=16),
    environment=dict(num_envs=4096),
    optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
)

algorithm(ippo, name="ippo")

board = dict(
    rows="${environment.kwargs.rows}",
    columns="${environment.kwargs.columns}",
    inarow="${environment.kwargs.inarow}",
)

connectx = dict(
    total_timesteps=20_000_000,
    environment=dict(num_envs=1024),
    rollout=dict(num_steps=32),
    algorithm=dict(
        num_minibatches=8, update_epochs=2, gamma=1.0, gae_lambda=0.95, entropy_coefficient=0.01
    ),
    optimizer=dict(lr=3e-4, max_grad_norm=0.5),
    network=dict(features=256, layers=2),
    score="gauntlet/mean",
    artisans=dict(
        gauntlet=fbuilds(
            Gauntlet,
            opponents=dict(
                random=builds(uniform, zen_partial=True),
                **{f"negamax_{depth}": builds(Negamax, depth=depth, **board) for depth in (1, 2, 4, 6)},
            ),
        )
    ),
)

special("ippo", "connectx", **connectx)
special("ippo", "jaxmarl", total_timesteps=20_000_000)
special(
    "ippo",
    "mapox",
    total_timesteps=20_000_000,
    environment=dict(num_envs=2048),
    rollout=dict(num_steps=16),
    algorithm=dict(
        num_minibatches=8, update_epochs=2, gamma=0.99, gae_lambda=0.95, entropy_coefficient=0.01
    ),
    optimizer=dict(lr=3e-4, max_grad_norm=0.5),
)

search(
    "ippo",
    "connectx",
    n_trials=1024,
    n_jobs=1,
    num_random_samples=16,
    resample_frequency=16,
    max_failure_rate=0.5,
    seed=0,
    max_suggestion_cost=900,
    params={
        "total_timesteps": dict(distribution="log_normal", min=5e7, max=2e9, center=7.14e8, scale=1.92),
        "environment.num_envs": dict(
            distribution="uniform_pow2", min=512, max=65536, center=16384, scale=8.33
        ),
        "rollout.num_steps": dict(distribution="uniform_pow2", min=2, max=32, center=8, scale=3.33),
        "algorithm.num_minibatches": dict(
            distribution="uniform_pow2", min=1, max=32, center=4, scale=5.0
        ),
        "algorithm.update_epochs": dict(distribution="int_uniform", min=1, max=8, center=6, scale=8.33),
        "network.features": dict(distribution="uniform_pow2", min=64, max=1024, center=512, scale=5.0),
        "network.layers": dict(distribution="int_uniform", min=1, max=4, center=2, scale=3.33),
        "optimizer.lr": dict(distribution="log_normal", min=1e-5, max=3e-3, center=1.46e-3, scale=3.61),
        "optimizer.max_grad_norm": dict(
            distribution="log_normal", min=0.05, max=5.0, center=2.44, scale=2.81
        ),
        "algorithm.entropy_coefficient": dict(
            distribution="log_normal", min=1e-5, max=0.5, center=0.0994, scale=6.66
        ),
        "algorithm.gae_lambda": dict(
            distribution="logit_normal", min=0.5, max=0.995, center=0.989, scale=3.26
        ),
        "algorithm.clip_coefficient": dict(
            distribution="uniform", min=0.1, max=0.5, center=0.155, scale=0.57
        ),
        "algorithm.value_coefficient": dict(
            distribution="log_normal", min=0.1, max=10.0, center=0.723, scale=1.9
        ),
    },
)
