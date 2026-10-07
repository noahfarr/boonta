import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from optax.contrib._muon import MuonDimensionNumbers
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_pupo import RecurrentPuPO
from boonta.environments.wrappers import (FlattenObservation,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, StickyAction,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def make(cfg):
    linear = (cfg.get("network") or {}).get("encoder") == "linear"
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)
    sticky = cfg.environment.get("sticky_action_prob", 0.0)
    if sticky:
        env = StickyAction(env, probability=sticky)
    if linear:
        env = FlattenObservation(env)
    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    torso = instantiate(cfg.stack)

    if linear:
        encoder = [
            lambda obs: obs.astype(jnp.float32),
            nn.Dense(hidden_dim, use_bias=False, dtype=dtype),
        ]
    else:
        encoder = [
            lambda obs: obs.astype(jnp.float32),
            nn.Conv(16, (3, 3), padding="VALID"),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(hidden_dim),
            nn.relu,
        ]

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(encoder),
        ),
        torso=torso,
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, use_bias=False, dtype=dtype)),
            critic=nn.Dense(1, use_bias=False, dtype=dtype),
        ),
    )

    learning_rate = cfg.optimizer.lr
    if cfg.optimizer.get("anneal"):
        batch = cfg.environment.num_envs * cfg.rollout.num_steps
        updates = int(cfg.total_timesteps) // batch
        learning_rate = optax.cosine_decay_schedule(
            learning_rate,
            updates * cfg.algorithm.update_epochs * cfg.algorithm.num_minibatches,
            alpha=cfg.optimizer.get("min_lr_ratio", 0.0),
        )

    return {
        "algorithm": RecurrentPuPO(
            cfg=instantiate(cfg.algorithm),
            importance_exponent=instantiate(cfg.importance_exponent),
            network=network,
            optimizer=optax.chain(
                optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
                optax.multi_transform(
                    {
                        "muon": optax.contrib.muon(
                            learning_rate,
                            muon_weight_dimension_numbers=MuonDimensionNumbers(
                                -2, -1
                            ),
                        ),
                        "adam": optax.adam(learning_rate),
                    },
                    lambda params: jax.tree.map(
                        lambda p: "muon" if p.ndim >= 2 else "adam", params
                    ),
                )
                if cfg.optimizer.get("name") == "muon"
                else optax.adam(learning_rate),
            ),
        ),
        "environment": env,
    }
