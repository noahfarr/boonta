import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (ClipAction, NormalizeObservation,
                                          RecordEpisodeStatistics)
from boonta.networks import ActorCritic, FeatureExtractor, Gaussian, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    action_dim, *_ = env.action_space().shape
    env = instantiate(cfg.environment.auto_reset, env)

    actor = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=lambda obs: obs["state"]
        ),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.swish,
                nn.Dense(256),
                nn.swish,
                nn.Dense(128),
                nn.swish,
                nn.Dense(2 * action_dim),
            ]
        ),
    )

    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=lambda obs: obs["privileged_state"]
        ),
        head=nn.Sequential(
            [
                nn.Dense(512),
                nn.swish,
                nn.Dense(256),
                nn.swish,
                nn.Dense(128),
                nn.swish,
                nn.Dense(1),
            ]
        ),
    )

    network = ActorCritic(actor=Gaussian(actor), critic=critic)

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
    env = ClipAction(env, low=-1.0, high=1.0)
    env = NormalizeObservation(env)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
