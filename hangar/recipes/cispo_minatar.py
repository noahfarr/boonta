import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.cispo import CISPO
from boonta.environments.wrappers import (GroupedAutoReset,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)

    num_actions = env.action_space().num_actions

    env = Vectorize(env, num_envs=cfg.environment.num_envs)
    env = GroupedAutoReset(
        env,
        num_steps=cfg.podracer.config.num_steps,
        group_size=cfg.algorithm.group_size,
    )

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda obs: obs.astype(jnp.float32),
                    nn.Conv(16, (3, 3), padding="VALID"),
                    nn.relu,
                    Flatten(start_dim=-3),
                    nn.Dense(128),
                    nn.relu,
                ]
            ),
        ),
        head=Categorical(nn.Dense(num_actions)),
    )

    algorithm = CISPO(
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
