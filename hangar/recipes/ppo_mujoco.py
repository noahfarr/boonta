import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, FeatureExtractor, Gaussian, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    action_dim, *_ = env.action_space().shape

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

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
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
