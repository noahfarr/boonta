from boonta.algorithms.recurrent_dqn import RecurrentDQNConfig

from ..sections import Optimizer, Replay, Rollout
from ..store import fbuilds
from . import algorithm, epsilon, recurrent

recurrent_dqn = algorithm(
    dict(
        defaults=[*recurrent, {"/buffer": "trajectory"}],
        algorithm=fbuilds(RecurrentDQNConfig, updates_per_step=32, gamma=0.99, tau=0.005),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=262_144),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="recurrent_dqn",
)

recurrent_dqn.hyperparameters("gymnax/minatar", total_timesteps=10_000_000)
