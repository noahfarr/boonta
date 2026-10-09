import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.reppo import REPPO
from boonta.environments.wrappers import (NormalizeObservation,
                                          NormalizeReward,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import FeatureExtractor, Network, SquashedGaussian
from boonta.networks.layers import Identity, Parameter


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)

    action_dim, *_ = env.action_space().shape

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    actor = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(512),
                    nn.RMSNorm(),
                    nn.swish,
                    nn.Dense(512),
                    nn.RMSNorm(),
                    nn.swish,
                    nn.Dense(512),
                    nn.RMSNorm(),
                    nn.swish,
                ]
            ),
        ),
        head=SquashedGaussian(nn.Dense(2 * action_dim)),
    )

    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(),
            action_extractor=Identity(),
        ),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.RMSNorm(),
                nn.swish,
                nn.Dense(512),
                nn.RMSNorm(),
                nn.swish,
                nn.Dense(512),
                nn.RMSNorm(),
                nn.swish,
                nn.Dense(512),
                nn.RMSNorm(),
                nn.swish,
                nn.Dense(cfg.algorithm.num_bins),
            ]
        ),
    )

    algorithm = REPPO(
        cfg=instantiate(cfg.algorithm, action_dim=action_dim),
        actor=actor,
        critic=critic,
        alpha=Parameter(),
        lagrangian=Parameter(),
        actor_optimizer=optax.adam(cfg.optimizer.actor_lr),
        critic_optimizer=optax.adam(cfg.optimizer.critic_lr),
        alpha_optimizer=optax.adam(cfg.optimizer.alpha_lr),
        lagrangian_optimizer=optax.adam(cfg.optimizer.lagrangian_lr),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
