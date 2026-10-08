from boonta.algorithms.pqn import PQNConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, epsilon

pqn = algorithm(
    dict(
        algorithm=fbuilds(PQNConfig, num_minibatches=4, update_epochs=1, gamma=0.99, q_lambda=0.65),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="pqn",
)

pqn.hyperparameters(
    "gymnax/minatar",
    total_timesteps=80_000_000,
    algorithm=dict(num_minibatches=16),
    environment=dict(num_envs=4096),
)
