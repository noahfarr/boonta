import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten

from . import schedules


def symbolic(obs):
    """One-hot NAVIX's symbolic grid: object tag, colour and state are categories, not magnitudes."""
    return jnp.concatenate(
        [jax.nn.one_hot(obs[..., channel], size) for channel, size in enumerate((16, 8, 4))], axis=-1
    )


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    symbolic,
                    Flatten(start_dim=-3),
                    nn.Dense(256),
                    nn.relu,
                    nn.Dense(hidden_dim),
                    nn.relu,
                ]
            ),
        ),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions)),
            critic=nn.Dense(1),
        ),
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
