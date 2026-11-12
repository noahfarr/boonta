from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def flatten(obs, time_limit):
    # The last channel is the raw step count in every cell; unscaled it drowns the 0/1 board.
    scale = jnp.ones(obs.shape[-1]).at[-1].set(1.0 / time_limit)
    return (obs * scale).reshape(*obs.shape[:-3], -1).astype(jnp.float32)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    time_limit = env.time_limit()

    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [partial(flatten, time_limit=time_limit), nn.Dense(256), nn.relu, nn.Dense(256), nn.relu]
            ),
        ),
        head=ActorCritic(actor=Categorical(nn.Dense(num_actions)), critic=nn.Dense(1)),
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
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
