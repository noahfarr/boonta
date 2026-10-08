from boonta.algorithms.recurrent_pqn import RecurrentPQNConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, epsilon, recurrent

recurrent_pqn = algorithm(
    dict(
        defaults=recurrent,
        algorithm=fbuilds(RecurrentPQNConfig, num_minibatches=4, update_epochs=1, gamma=0.99, q_lambda=0.65),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="recurrent_pqn",
)

recurrent_pqn.hyperparameters("gymnax/minatar", total_timesteps=10_000_000)
