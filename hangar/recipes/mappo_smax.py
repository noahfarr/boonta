import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordMultiAgentEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    env = environments.make(**cfg.environment)
    num_agents, *_ = env.action_space().shape
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    actor = Network(
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
        head=nn.vmap(
            nn.Dense,
            in_axes=-2,
            out_axes=-2,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(num_actions),
    )

    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    Flatten(start_dim=-2),
                    nn.Dense(128),
                    nn.relu,
                    nn.Dense(128),
                    nn.relu,
                ]
            ),
        ),
        head=nn.Sequential(
            [
                nn.Dense(1),
                lambda value: jnp.broadcast_to(
                    value[..., None, :], (*value.shape[:-1], num_agents, 1)
                ),
            ]
        ),
    )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=ActorCritic(actor=Categorical(actor), critic=critic),
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
