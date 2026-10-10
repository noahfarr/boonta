import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.pqn import PQN
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import EpsilonGreedy, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    epsilon_schedule = optax.linear_schedule(
        cfg.exploration.start,
        cfg.exploration.end,
        int(cfg.total_timesteps * cfg.exploration.fraction),
    )

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda obs: obs.astype(jnp.float32),
                    nn.Conv(16, (3, 3), padding="VALID"),
                    nn.LayerNorm(),
                    nn.relu,
                    Flatten(start_dim=-3),
                    nn.Dense(128),
                    nn.LayerNorm(),
                    nn.relu,
                ]
            ),
        ),
        head=EpsilonGreedy(nn.Dense(num_actions)),
    )

    algorithm = PQN(
        cfg=instantiate(cfg.algorithm),
        network=network,
        exploration_schedule=epsilon_schedule,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
