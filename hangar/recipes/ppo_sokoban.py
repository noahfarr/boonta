import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Stagger,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def flatten(obs):
    return obs.reshape(*obs.shape[:-3], -1).astype(jnp.float32)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions

    env = SameStepAutoReset(env)
    env = Stagger(env, spread=120)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [flatten, nn.Dense(256), nn.relu, nn.Dense(256), nn.relu]
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
