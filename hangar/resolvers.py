import operator

import jax
import jax.numpy as jnp
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def get_action_dim(cfg):
    from boonta import environments

    env = environments.make(**cfg)
    space = env.action_space()

    if jnp.issubdtype(space.dtype, jnp.integer):
        return space.num_actions

    action_dim, *_ = space.shape
    return action_dim


def cascade(group: str, algorithm: str, environment: str, curriculum: str = None) -> str:
    loader = GlobalHydra.instance().config_loader()

    def offers(path):
        parent, leaf = path.rsplit("/", 1)
        return leaf in loader.get_group_options(f"{group}/{parent}")

    parts = environment.split("/")
    while parts:
        path = f"{algorithm}/{'/'.join(parts)}"

        if curriculum is not None and offers(f"{path}/{curriculum}"):
            return f"{path}/{curriculum}"
        if offers(path):
            return path

        parts.pop()

    return algorithm


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
OmegaConf.register_new_resolver("cascade", cascade)
OmegaConf.register_new_resolver("trial", trial)
OmegaConf.register_new_resolver("get_group", get_group)
OmegaConf.register_new_resolver("groups", groups)
OmegaConf.register_new_resolver("device_count", jax.device_count)
