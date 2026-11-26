import math

import jax.numpy as jnp
import optax
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.eggroll import EGGROLL
from boonta.environments.wrappers import Group, RecordEpisodeStatistics
from boonta.networks.low_rank_perturbations import LowRankPerturbedMLP


def make(cfg):
    choices = HydraConfig.get().runtime.choices
    assert choices["podracer"] == "anakin", "EGGROLL requires Anakin"
    assert choices["curriculum"] == "default", "EGGROLL requires the default curriculum"
    assert (
        choices["environment/auto_reset"] == "same_step"
    ), "EGGROLL requires same-step auto-reset"

    env = environments.make(**cfg.environment)
    action_space = env.action_space()
    if jnp.issubdtype(action_space.dtype, jnp.integer):
        if action_space.shape:
            raise ValueError(
                "EGGROLL requires a scalar discrete or continuous action space"
            )
        output_size = action_space.num_actions
    else:
        output_size = math.prod(action_space.shape)

    algorithm = EGGROLL(
        cfg=instantiate(cfg.algorithm),
        network=LowRankPerturbedMLP(
            widths=tuple(cfg.network.hidden_sizes) + (output_size,),
        ),
        action_space=action_space,
        optimizer=optax.adam(cfg.optimizer.lr),
    )
    env = instantiate(cfg.environment.auto_reset, env)
    env = Group(env, num_steps=cfg.podracer.config.num_steps)
    env = RecordEpisodeStatistics(env, gamma=1.0)
    return {"algorithm": algorithm, "environment": env}
