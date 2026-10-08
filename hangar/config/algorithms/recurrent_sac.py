from boonta.algorithms.recurrent_sac import RecurrentSACConfig

from ..sections import Optimizers, Replay, Rollout
from ..store import fbuilds
from . import algorithm

algorithm(
    dict(
        defaults=[{"/torso": "default"}, {"/cell": "min_gru"}, {"/stack": "default"}, {"/buffer": "trajectory"}],
        algorithm=fbuilds(
            RecurrentSACConfig,
            updates_per_step=64,
            gamma=0.997,
            tau=0.005,
            zen_exclude=("target_entropy",),
        ),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=1_048_576),
        optimizer=Optimizers(actor_lr=6e-4, critic_lr=6e-4, alpha_lr=6e-4),
    ),
    name="recurrent_sac",
)
