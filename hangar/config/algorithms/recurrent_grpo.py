from boonta.algorithms.recurrent_grpo import RecurrentGRPOConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm

recurrent_grpo = algorithm(
    dict(
        algorithm=fbuilds(
            RecurrentGRPOConfig,
            group_size=8,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.04,
            gamma=1.0,
        ),
        rollout=Rollout(num_steps=30),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=1e-5, max_grad_norm=0.5),
    ),
    name="recurrent_grpo",
)

recurrent_grpo.hyperparameters(
    "wordle",
    environment=dict(num_envs=32),
    network=dict(repo_id="Qwen/Qwen3-0.6B-Base", dtype="bfloat16", param_dtype="bfloat16"),
)
