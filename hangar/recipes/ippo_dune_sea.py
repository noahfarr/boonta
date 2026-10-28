from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordMultiAgentEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten
from boonta.utils.typing import Dtype


class ViewAndEnergy(nn.Module):
    features: int = 128
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        dtype = self.dtype or jnp.float32
        view = Flatten(start_dim=-3)(obs["view"].astype(dtype))
        energy = obs["energy"].astype(dtype)[..., None]
        inventory = jnp.log1p(obs["inventory"].astype(jnp.float32)).astype(dtype)[..., None]
        team = obs["team"].astype(dtype)[..., None]
        dense = partial(nn.Dense, self.features, dtype=self.dtype, param_dtype=self.param_dtype)
        return nn.Sequential([dense(), nn.relu, dense(), nn.relu])(
            jnp.concatenate([view, energy, inventory, team], axis=-1)
        )


def make(cfg):
    env = environments.make(**cfg.environment)
    dtype = (cfg.get("network") or {}).get("dtype")
    dtype = jnp.dtype(dtype) if dtype else None

    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=ViewAndEnergy(dtype=dtype),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, dtype=dtype)),
            critic=nn.Dense(1, dtype=dtype),
        ),
    )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
