import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (RecordMultiAgentEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)

    num_actions = env.action_space().num_actions

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(128),
                    nn.relu,
                    nn.Dense(128),
                    nn.relu,
                ]
            ),
        ),
        head=ActorCritic(
            actor=Categorical(
                nn.vmap(
                    nn.Dense,
                    in_axes=-2,
                    out_axes=-2,
                    variable_axes={"params": None},
                    split_rngs={"params": False},
                )(num_actions)
            ),
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
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
