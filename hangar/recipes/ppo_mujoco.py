import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (ClipAction, NormalizeObservation,
                                          NormalizeReward,
                                          RecordEpisodeStatistics)
from boonta.networks import ActorCritic, FeatureExtractor, Gaussian, Network
from boonta.networks.layers import Identity

from . import schedules
from .separate_actor_critic import StateIndependentScale, mlp


def make(cfg):
    env = environments.make(**cfg.environment)
    action_dim, *_ = env.action_space().shape
    env = instantiate(cfg.environment.auto_reset, env)

    settings = cfg.get("network") or {}
    if settings.get("architecture", "shared") == "separate":
        width, activation = settings.get("width", 256), settings.get("activation", "tanh")
        network = Network(
            feature_extractor=FeatureExtractor(observation_extractor=Identity()),
            head=ActorCritic(
                actor=Gaussian(
                    StateIndependentScale(mlp(width, activation, action_dim, 0.01), action_dim)
                ),
                critic=mlp(width, activation, 1, 1.0),
            ),
        )
    else:
        network = Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential(
                    [
                        nn.Dense(256),
                        nn.tanh,
                        nn.Dense(256),
                        nn.tanh,
                    ]
                ),
            ),
            head=ActorCritic(
                actor=Gaussian(nn.Dense(2 * action_dim)),
                critic=nn.Dense(1),
            ),
        )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(
                schedules.learning_rate(
                    cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
                ),
                eps=cfg.optimizer.get("eps", 1e-8),
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    if cfg.environment.get("normalize"):
        env = NormalizeReward(NormalizeObservation(ClipAction(env)), gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
