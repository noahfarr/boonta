import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_dqn import RecurrentDQN
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import EpsilonGreedy, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    epsilon_schedule = optax.linear_schedule(
        cfg.exploration.start,
        cfg.exploration.end,
        int(cfg.total_timesteps * cfg.exploration.fraction),
    )

    torso = instantiate(cfg.stack)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda obs: obs.astype(jnp.float32),
                    nn.Conv(16, (3, 3), padding="VALID"),
                    nn.LayerNorm(),
                    nn.relu,
                    Flatten(start_dim=-3),
                    nn.Dense(hidden_dim),
                    nn.LayerNorm(),
                    nn.relu,
                ]
            ),
        ),
        torso=torso,
        head=EpsilonGreedy(nn.Dense(num_actions)),
    )

    return {
        "algorithm": RecurrentDQN(
            cfg=instantiate(cfg.algorithm),
            network=network,
            exploration_schedule=epsilon_schedule,
            buffer=instantiate(cfg.buffer),
            optimizer=optax.chain(
                optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
                optax.adam(cfg.optimizer.lr),
            ),
        ),
        "environment": env,
    }
