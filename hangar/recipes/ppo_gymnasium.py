import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import (ActorCritic, Categorical, FeatureExtractor,
                             Gaussian, Network)


def make(cfg):
    env = environments.make(**cfg.environment)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    space = env.action_space()
    if jnp.issubdtype(space.dtype, jnp.integer):
        actor = Categorical(nn.Dense(space.num_actions))
    else:
        action_dim, *_ = space.shape
        actor = Gaussian(nn.Dense(2 * action_dim))

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
        head=ActorCritic(actor=actor, critic=nn.Dense(1)),
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
