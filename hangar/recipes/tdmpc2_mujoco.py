import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.tdmpc2 import TDMPC2
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import FeatureExtractor, Network, SquashedGaussian
from boonta.networks.layers import Identity, SimNorm
from boonta.planners import MPC


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    action_dim, *_ = env.action_space().shape

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    encoder = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(512),
                    nn.LayerNorm(),
                    jax.nn.mish,
                    nn.Dense(512),
                    nn.LayerNorm(),
                    jax.nn.mish,
                    nn.Dense(512),
                    SimNorm(dim=8),
                ]
            ),
        ),
    )

    dynamics = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(),
            action_extractor=Identity(),
        ),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(512),
                SimNorm(dim=8),
            ]
        ),
    )

    reward = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(),
            action_extractor=Identity(),
        ),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(cfg.algorithm.num_bins),
            ]
        ),
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
            axis_size=cfg.algorithm.num_critics,
        )(
            [
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(cfg.algorithm.num_bins),
            ]
        ),
    )

    actor = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(512),
                    nn.LayerNorm(),
                    jax.nn.mish,
                    nn.Dense(512),
                    nn.LayerNorm(),
                    jax.nn.mish,
                ]
            ),
        ),
        head=SquashedGaussian(nn.Dense(2 * action_dim)),
    )

    termination = Network(
        feature_extractor=FeatureExtractor(observation_extractor=Identity()),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(512),
                nn.LayerNorm(),
                jax.nn.mish,
                nn.Dense(1),
            ]
        ),
    )

    controller = MPC(
        planner=instantiate(
            cfg.planner,
            action_shape=(cfg.environment.num_envs, cfg.algorithm.horizon, action_dim),
        )
    )

    return {
        "algorithm": TDMPC2(
            cfg=instantiate(cfg.algorithm, action_dim=action_dim),
            encoder=encoder,
            dynamics=dynamics,
            reward=reward,
            critic=critic,
            actor=actor,
            termination=termination,
            controller=controller,
            buffer=instantiate(cfg.buffer),
            world_model_optimizer=optax.adam(cfg.optimizer.world_model_lr),
            actor_optimizer=optax.adam(cfg.optimizer.actor_lr),
        ),
        "environment": env,
    }
