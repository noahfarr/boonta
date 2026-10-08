import operator

import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from hangar.config.algorithms import cascade, spaces, table


def get_action_dim(cfg):
    from boonta import environments

    env = environments.make(**cfg)
    space = env.action_space()

    if jnp.issubdtype(space.dtype, jnp.integer):
        return space.num_actions

    action_dim, *_ = space.shape
    return action_dim


def tuned(algorithm: str, environment: str) -> str:
    return cascade(table, algorithm, environment)


def search_space(algorithm: str, environment: str) -> str:
    return cascade(spaces, algorithm, environment)


def trial():
    return OmegaConf.select(HydraConfig.get(), "job.num", default=0)


def get_group(_root_):
    algorithm = HydraConfig.get().runtime.choices["algorithm"]
    group = f"{algorithm}_{_root_.environment.namespace}_{_root_.environment.env_id}"
    return group[:128]


def groups():
    choices = HydraConfig.get().runtime.choices
    return {k: v for k, v in choices.items() if not k.startswith("hydra/")}


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("metric", lambda name: operator.itemgetter(name))
OmegaConf.register_new_resolver("get_action_dim", get_action_dim)
OmegaConf.register_new_resolver("hyperparameters", tuned)
OmegaConf.register_new_resolver("search_space", search_space)
OmegaConf.register_new_resolver("trial", trial)
OmegaConf.register_new_resolver("get_group", get_group)
OmegaConf.register_new_resolver("groups", groups)
