import operator

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (GroupedAutoReset,
                                          RecordMultiAgentEpisodeStatistics,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten

MASKED_LOGIT = -1e9


class View(nn.Module):
    sizes: tuple[int, ...]
    features: int = 128

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        view = obs["view"].astype(jnp.int32)
        planes = jnp.concatenate(
            [
                jax.nn.one_hot(view[..., channel], size)
                for channel, size in enumerate(self.sizes)
            ],
            axis=-1,
        )
        state = nn.Sequential(
            [
                Flatten(start_dim=-3),
                nn.Dense(self.features),
                nn.relu,
                nn.Dense(self.features),
                nn.relu,
            ]
        )(planes)
        return {"state": state, "mask": obs["mask"]}


class Masked(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.num_actions)(x["state"])
        return jnp.where(x["mask"], logits, MASKED_LOGIT)


def make(cfg):
    env = environments.make(**cfg.environment)

    num_actions = env.action_space().num_actions
    sizes = tuple(int(size) for size in env.observation_space()["view"].high + 1)
    time_limit = env.time_limit()

    env = Vectorize(env, num_envs=cfg.environment.num_envs)
    env = GroupedAutoReset(env, num_steps=time_limit)
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    network = Network(
        feature_extractor=View(sizes=sizes),
        head=ActorCritic(
            actor=Categorical(Masked(num_actions)),
            critic=nn.Sequential([operator.itemgetter("state"), nn.Dense(1)]),
        ),
    )

    return {
        "algorithm": PPO(
            cfg=instantiate(cfg.algorithm),
            network=network,
            optimizer=optax.chain(
                optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
                optax.adam(cfg.optimizer.lr),
            ),
        ),
        "environment": env,
    }
