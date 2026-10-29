from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import (LogInfo,
                                             RecordMultiAgentEpisodeStatistics)
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten
from boonta.utils.typing import Dtype

from . import schedules


class ViewAndEnergy(nn.Module):
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        dtype = self.dtype or jnp.float32
        view = Flatten(start_dim=-3)(obs["view"].astype(dtype))
        energy = obs["energy"].astype(dtype)[..., None]
        inventory = jnp.log1p(obs["inventory"].astype(jnp.float32)).astype(dtype)[..., None]
        thief = obs["thief"].astype(dtype)[..., None]
        storm = obs["storm"].astype(dtype)[..., None]
        dense = partial(nn.Dense, self.features, dtype=self.dtype, param_dtype=self.param_dtype)
        return nn.Sequential([dense(), nn.relu])(
            jnp.concatenate([view, energy, inventory, thief, storm], axis=-1)
        )


def make(cfg):
    env = environments.make(**cfg.environment)
    dtype = (cfg.get("network") or {}).get("dtype")
    dtype = jnp.dtype(dtype) if dtype else None

    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features
    num_agents = env.num_agents
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=ViewAndEnergy(features=hidden_dim, dtype=dtype),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, dtype=dtype)),
            critic=nn.Dense(1, dtype=dtype),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg,
        batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps * num_agents,
    )

    algorithm = RecurrentPPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = LogInfo(env, keys=("tags", "stolen", "sheltered", "exposed"), tags=())
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
