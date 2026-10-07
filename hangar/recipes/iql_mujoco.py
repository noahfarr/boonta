import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import datasets
from boonta import environments
from boonta.algorithms.iql import IQL
from boonta.environments.wrappers import RecordEpisodeStatistics, SameStepAutoReset, Vectorize
from boonta.networks import FeatureExtractor, Gaussian, Network
from boonta.networks.layers import Identity


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

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
        head=Gaussian(nn.Dense(2 * action_dim)),
    )

    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(),
            action_extractor=Identity(),
        ),
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

    value = Network(
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
        head=nn.Dense(1),
    )

    return {
        "algorithm": IQL(
            cfg=instantiate(cfg.algorithm),
            actor=actor,
            critic=critic,
            value=value,
            actor_optimizer=optax.adam(cfg.optimizer.actor_lr),
            critic_optimizer=optax.adam(cfg.optimizer.critic_lr),
            value_optimizer=optax.adam(cfg.optimizer.value_lr),
        ),
        "environment": env,
        "dataset": datasets.make(**cfg.dataset),
    }
