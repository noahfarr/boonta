import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import orthogonal
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules

WIDTH = 64


def trunk(output: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        [
            nn.Dense(WIDTH, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Dense(WIDTH, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            output,
        ]
    )


def features(num_actions: int) -> FeatureExtractor:
    return FeatureExtractor(
        observation_extractor=nn.Sequential(
            [nn.Dense(WIDTH, kernel_init=orthogonal(np.sqrt(2))), nn.relu]
        ),
        action_extractor=nn.Embed(num_actions, WIDTH),
    )


def head(num_actions: int) -> ActorCritic:
    return ActorCritic(
        actor=Categorical(trunk(nn.Dense(num_actions, kernel_init=orthogonal(0.01)))),
        critic=trunk(nn.Dense(1, kernel_init=orthogonal(1.0))),
    )


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=features(num_actions),
        torso=instantiate(cfg.stack),
        head=head(num_actions),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
    )

    algorithm = RecurrentPPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
