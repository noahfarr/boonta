import math

import jax.numpy as jnp
import optax
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.eggroll import EGGROLL
from boonta.environments.wrappers import Group, RecordEpisodeStatistics, TransformAction
from boonta.networks import Categorical, SquashedGaussian
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
    widths = tuple(cfg.network.hidden_sizes)
    if jnp.issubdtype(action_space.dtype, jnp.integer):
        network = Categorical(LowRankPerturbedMLP(widths + (action_space.num_actions,)))
    else:
        low, high = action_space.low, action_space.high
        network = SquashedGaussian(
            LowRankPerturbedMLP(widths + (2 * math.prod(action_space.shape),))
        )
        env = TransformAction(env, lambda action: low + (action + 1.0) * 0.5 * (high - low))

    algorithm = EGGROLL(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.adam(cfg.optimizer.lr),
    )
    env = instantiate(cfg.environment.auto_reset, env)
    env = Group(env, num_steps=cfg.podracer.config.num_steps)
    env = RecordEpisodeStatistics(env, gamma=1.0)
    return {"algorithm": algorithm, "environment": env}
