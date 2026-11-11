from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics, Stagger
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def flatten(obs):
    return obs.reshape(*obs.shape[:-3], -1).astype(jnp.float32)


def one_hot_flatten(obs, time_limit=120):
    """Categorical board layers and one scalar clock for feed-forward PPO."""
    variable = jax.nn.one_hot(obs[..., 0].astype(jnp.int32), 5)
    fixed = jax.nn.one_hot(obs[..., 1].astype(jnp.int32), 3)
    grid = jnp.concatenate([variable, fixed], axis=-1)
    grid = grid.reshape((*obs.shape[:-3], -1))
    clock = obs[..., 0, 0, 2:3].astype(jnp.float32) / time_limit
    return jnp.concatenate([grid, clock], axis=-1)


def puffer_planes(obs):
    """Puffer Boxoban's channel-first binary planes, without a clock."""
    variable, fixed = obs[..., 0], obs[..., 1]
    planes = jnp.stack(
        [variable == 3, fixed == 1, variable == 4, fixed == 2], axis=-3
    )
    return planes.reshape((*obs.shape[:-3], -1)).astype(jnp.float32)


class PufferEncoder(nn.Module):
    features: int = 1024

    @nn.compact
    def __call__(self, obs):
        return nn.Dense(self.features, use_bias=False)(puffer_planes(obs))


def cnn_planes(obs, time_limit=120):
    planes = puffer_planes(obs).reshape((*obs.shape[:-3], 4, *obs.shape[-3:-1]))
    planes = jnp.moveaxis(planes, -3, -1)
    clock = obs[..., 2:3].astype(jnp.float32) / time_limit
    return jnp.concatenate([planes, clock], axis=-1)


class CNNEncoder(nn.Module):
    features: int = 256
    time_limit: int = 120

    @nn.compact
    def __call__(self, obs):
        hidden = cnn_planes(obs, self.time_limit)
        for channels in (32, 64):
            hidden = nn.relu(nn.Conv(channels, (3, 3), padding='SAME')(hidden))
        hidden = hidden.reshape((*obs.shape[:-3], -1))
        return nn.relu(nn.Dense(self.features)(hidden))


def observation_extractor(encoder='one_hot', features=256, time_limit=120):
    if encoder not in ('one_hot', 'cnn', 'puffer', 'flat'):
        raise ValueError(f'Unknown Sokoban encoder: {encoder}')
    if encoder == 'one_hot':
        first = partial(one_hot_flatten, time_limit=time_limit)
    elif encoder == 'cnn':
        first = CNNEncoder(features=features, time_limit=time_limit)
    elif encoder == 'puffer':
        first = PufferEncoder(features=features)
    else:
        first = flatten
    # Retain the existing nonlinear feed-forward body after the input encoder.
    return nn.Sequential([first, nn.Dense(256), nn.relu, nn.Dense(256), nn.relu])


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    time_limit = env.time_limit()

    env = instantiate(cfg.environment.auto_reset, env)
    env = Stagger(env, spread=time_limit)

    net_cfg = cfg.get('network') or {}
    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=observation_extractor(
                encoder=net_cfg.get('encoder', 'one_hot'),
                features=net_cfg.get('encoder_features', 256),
                time_limit=time_limit,
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
