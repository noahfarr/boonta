import operator

import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (RecordMultiAgentEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten

MASKED_LOGIT = -1e9


class Board(nn.Module):
    features: int = 256
    layers: int = 2

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        state = Flatten(start_dim=-3)(obs["board"])
        for _ in range(self.layers):
            state = nn.relu(nn.Dense(self.features)(state))
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

    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=Board(
            features=cfg.network.features, layers=cfg.network.layers
        ),
        head=ActorCritic(
            actor=Categorical(Masked(num_actions)),
            critic=nn.Sequential([operator.itemgetter("state"), nn.Dense(1)]),
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
