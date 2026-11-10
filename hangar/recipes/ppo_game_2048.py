import operator

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics, TimeLimit, TransformReward
from boonta.networks import ActorCritic, Categorical, Network

from . import schedules

MASKED_LOGIT = -1e9


class Board(nn.Module):
    """PufferLib's 2048 encoder: the 16 tile exponents as floats into one linear layer without bias.

    PufferLib follows it with a MinGRU stack; this recipe stays feedforward with dense layers instead.
    """

    features: int = 256
    layers: int = 2

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        board = obs["board"]
        state = board.reshape(*board.shape[:-2], -1).astype(jnp.float32)
        state = nn.Dense(self.features, use_bias=False)(state)
        for _ in range(self.layers):
            state = nn.relu(nn.Dense(self.features, use_bias=False)(state))
        return {"state": state, "mask": obs["mask"]}


class Masked(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.num_actions, use_bias=False)(x["state"])
        return jnp.where(x["mask"], logits, MASKED_LOGIT)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions

    # Jumanji pays the value of every merged tile, which grows to thousands. A merge into 2**k pays
    # about k after log2, close to PufferLib's slowly growing per-merge rewards.
    env = TransformReward(env, lambda reward: jnp.log2(1.0 + reward))
    # Jumanji's 2048 runs until the board locks. Masked moves always end it, but evaluation needs a
    # longest episode to size its rollout, so long games are truncated.
    env = TimeLimit(env, time_limit=cfg.environment.get("time_limit", 10_000))
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=Board(features=cfg.network.features, layers=cfg.network.layers),
        head=ActorCritic(
            actor=Categorical(Masked(num_actions)),
            critic=nn.Sequential(
                [operator.itemgetter("state"), nn.Dense(1, use_bias=False)]
            ),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
    )
    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.contrib.muon(
                learning_rate,
                beta=cfg.optimizer.get("beta", 0.95),
                muon_weight_dimension_numbers=lambda params: jax.tree.map(
                    lambda p: MuonDimensionNumbers(-2, -1) if p.ndim >= 2 else None,
                    params,
                ),
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
