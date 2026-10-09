import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (DomainRandomization,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def randomize_gravity(key, params):
    gravity = jax.random.uniform(key, minval=5.0, maxval=20.0)
    return params.replace(gravity=gravity)


def make(cfg):
    env = environments.make(**cfg.environment)
    env = DomainRandomization(env, randomize_gravity)
    env = SameStepAutoReset(env)

    num_actions = env.action_space().num_actions

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(64),
                    nn.tanh,
                    nn.Dense(64),
                    nn.tanh,
                ]
            ),
        ),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions)),
            critic=nn.Dense(1),
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
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
