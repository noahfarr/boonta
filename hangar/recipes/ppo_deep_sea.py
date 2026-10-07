import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (FlattenObservation,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, Stagger,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    env = FlattenObservation(env)
    num_actions = env.action_space().num_actions

    env = SameStepAutoReset(env)
    env = Stagger(env, spread=cfg.environment.kwargs.size)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
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

    return {
        "algorithm": PPO(
            cfg=instantiate(cfg.algorithm),
            network=network,
            optimizer=optax.chain(
                optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
                optax.adam(cfg.optimizer.lr),
            ),
        ),
        "environment": env,
    }
