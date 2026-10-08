import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.recurrent_pupo import RecurrentPuPO
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Stagger,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules


class Board(nn.Module):
    features: int
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, obs):
        grid, clock = obs[..., :2], obs[..., 2:3]
        planes = jax.nn.one_hot(grid.astype(jnp.int32), 7, dtype=self.dtype or jnp.float32)
        planes = planes.reshape(*grid.shape[:-3], *grid.shape[-3:-1], -1)
        planes = jnp.concatenate(
            [planes, clock.astype(planes.dtype) / 120.0], axis=-1
        )
        hidden = nn.Conv(32, (3, 3), use_bias=False, dtype=self.dtype)(planes)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(64, (3, 3), use_bias=False, dtype=self.dtype)(hidden)
        hidden = nn.relu(hidden)
        hidden = hidden.reshape(*hidden.shape[:-3], -1)
        return nn.Dense(self.features, use_bias=False, dtype=self.dtype)(hidden)


class Flat(nn.Module):
    features: int
    depth: int = 3
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, obs):
        grid, clock = obs[..., :2], obs[..., 2:3]
        planes = jax.nn.one_hot(grid.astype(jnp.int32), 7, dtype=self.dtype or jnp.float32)
        hidden = jnp.concatenate(
            [
                planes.reshape(*grid.shape[:-3], -1),
                clock.reshape(*grid.shape[:-3], -1).astype(planes.dtype) / 120.0,
            ],
            axis=-1,
        )
        for _ in range(self.depth):
            hidden = nn.relu(nn.Dense(self.features, use_bias=False, dtype=self.dtype)(hidden))
        return hidden


def make(cfg):
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features

    env = SameStepAutoReset(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    env = Stagger(env, spread=120)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    encoders = {"conv": Board, "flat": Flat}
    encoder = encoders[cfg.get("encoder", "conv")]
    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=encoder(features=hidden_dim, dtype=dtype),
        ),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, use_bias=False, dtype=dtype)),
            critic=nn.Dense(1, use_bias=False, dtype=dtype),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.rollout.num_steps
    )

    algorithm = RecurrentPuPO(
        cfg=instantiate(cfg.algorithm),
        importance_exponent=instantiate(cfg.importance_exponent),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.multi_transform(
                {
                    "muon": optax.contrib.muon(
                        learning_rate,
                        muon_weight_dimension_numbers=MuonDimensionNumbers(-2, -1),
                    ),
                    "adam": optax.adam(learning_rate),
                },
                lambda params: jax.tree.map(
                    lambda p: "muon" if p.ndim >= 2 else "adam", params
                ),
            )
            if cfg.optimizer.get("name") == "muon"
            else optax.adam(
                learning_rate,
                b1=cfg.optimizer.get("b1", 0.9),
                b2=cfg.optimizer.get("b2", 0.999),
                eps=cfg.optimizer.get("eps", 1e-8),
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
