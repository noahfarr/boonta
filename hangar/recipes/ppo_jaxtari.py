from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    dtype = (cfg.get("network") or {}).get("dtype")
    conv = partial(nn.Conv, padding="VALID", dtype=dtype)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda obs: jnp.moveaxis(obs[..., 0], -3, -1).astype(jnp.float32)
                    / 255.0,
                    conv(32, (8, 8), strides=(4, 4)),
                    nn.relu,
                    conv(64, (4, 4), strides=(2, 2)),
                    nn.relu,
                    conv(64, (3, 3), strides=(1, 1)),
                    nn.relu,
                    Flatten(start_dim=-3),
                    nn.Dense(512, dtype=dtype),
                    nn.relu,
                ]
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
