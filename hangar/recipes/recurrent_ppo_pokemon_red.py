import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.peanut_gb import pokemon_red
from boonta.environments.wrappers import (LogInfo, RecordEpisodeStatistics,
                                          SameStepAutoReset, Stagger,
                                          TimeLimit, Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules


class NatureCNN(nn.Module):
    features: int
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, obs):
        dtype = self.dtype or jnp.float32
        hidden = jnp.moveaxis(obs, -3, -1).astype(dtype) * jnp.asarray(1.0 / 255.0, dtype)
        hidden = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID", dtype=self.dtype)(hidden))
        hidden = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID", dtype=self.dtype)(hidden))
        hidden = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID", dtype=self.dtype)(hidden))
        hidden = hidden.reshape(hidden.shape[:-3] + (-1,))
        return nn.relu(nn.Dense(self.features, dtype=self.dtype)(hidden))


class Senses(nn.Module):
    features: int
    tiles: int = 128
    vectors: int = 128
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, obs):
        seen = NatureCNN(self.tiles, dtype=self.dtype)(obs["tiles"])
        image = NatureCNN(self.features, dtype=self.dtype)(obs["image"])
        senses = [image, seen]
        for name in ("flags", "vitals"):
            held = obs[name].astype(self.dtype or jnp.float32)
            senses.append(nn.relu(nn.Dense(self.vectors, dtype=self.dtype)(held)))
        return jnp.concatenate(senses, axis=-1)


def make(cfg):
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)
    time_limit = env.time_limit()
    env = TimeLimit(env, time_limit)
    env = SameStepAutoReset(env)
    env = Stagger(env, spread=time_limit)
    num_actions = env.action_space().num_actions
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Senses(features=cfg.cell.features, dtype=dtype)
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

    algorithm = RecurrentPPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.contrib.muon(
                learning_rate,
                beta=cfg.optimizer.beta,
                weight_decay=cfg.optimizer.weight_decay,
                muon_weight_dimension_numbers=lambda params: jax.tree.map(
                    lambda p: MuonDimensionNumbers(-2, -1) if p.ndim >= 2 else None,
                    params,
                ),
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = LogInfo(env, keys=pokemon_red.KEYS)
    env = pokemon_red.LogFlags(env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
