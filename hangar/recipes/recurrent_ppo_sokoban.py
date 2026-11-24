import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import RecordEpisodeStatistics, Stagger
from boonta.networks import (SSM, ActorCritic, Categorical, FeatureExtractor,
                             Highway, MinGRUCell, Network, Tower)

from . import schedules

FEATURES = 256


class Board(nn.Module):
    features: int
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, obs):
        grid, clock = obs[..., :2], obs[..., 2:3]
        planes = jax.nn.one_hot(grid.astype(jnp.int32), 7, dtype=self.dtype or jnp.float32)
        planes = planes.reshape(*grid.shape[:-3], *grid.shape[-3:-1], -1)
        planes = jnp.concatenate([planes, clock.astype(planes.dtype) / 120.0], axis=-1)
        hidden = nn.relu(nn.Conv(32, (3, 3), use_bias=False, dtype=self.dtype)(planes))
        hidden = nn.relu(nn.Conv(64, (3, 3), use_bias=False, dtype=self.dtype)(hidden))
        hidden = hidden.reshape(*hidden.shape[:-3], -1)
        return nn.Dense(self.features, use_bias=False, dtype=self.dtype)(hidden)


def make(cfg):
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)
    env = Stagger(env, spread=120)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Board(features=FEATURES, dtype=dtype),
        ),
        torso=Tower(
            Highway(SSM(MinGRUCell(features=FEATURES, dtype=dtype)), dtype=dtype),
            num_layers=2,
        ),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, use_bias=False, dtype=jnp.float32)),
            critic=nn.Dense(1, use_bias=False, dtype=jnp.float32),
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
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
