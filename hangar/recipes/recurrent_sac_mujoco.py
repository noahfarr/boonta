import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_sac import RecurrentSAC
from boonta.environments.wrappers import (RecordEpisodeStatistics,
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
                    nn.Dense(256),
                    nn.relu,
                    nn.Dense(256),
                    nn.relu,
                ]
            ),
        ),
        torso=instantiate(cfg.stack),
        head=SquashedGaussian(nn.Dense(2 * action_dim)),
    )

    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(),
            action_extractor=Identity(),
        ),
        torso=instantiate(cfg.stack),
        head=nn.vmap(
            nn.Sequential,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=2,
        )(
            [
                nn.Dense(256),
                nn.relu,
                nn.Dense(256),
                nn.relu,
                nn.Dense(1),
            ]
        ),
    )

    algorithm = RecurrentSAC(
        cfg=instantiate(cfg.algorithm, target_entropy=-float(action_dim)),
        actor=actor,
        critic=critic,
        alpha=Parameter(),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=optax.adam(cfg.optimizer.actor_lr),
        critic_optimizer=optax.adam(cfg.optimizer.critic_lr),
        alpha_optimizer=optax.adam(cfg.optimizer.alpha_lr),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
