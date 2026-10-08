from boonta.algorithms.dqn import DQNConfig

from ..sections import Optimizer, Replay, Rollout
from ..store import fbuilds
from . import algorithm, epsilon

dqn = algorithm(
    dict(
        defaults=[{"/buffer": "transition"}],
        algorithm=fbuilds(DQNConfig, updates_per_step=32, gamma=0.99, tau=0.005),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=262_144),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="dqn",
)

dqn.hyperparameters("gymnax/minatar", total_timesteps=10_000_000)
