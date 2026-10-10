import operator

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from optax.contrib._muon import MuonDimensionNumbers
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_pupo import RecurrentPuPO
from boonta.environments.wrappers import (Group,
                                          RecordMultiAgentEpisodeStatistics,
                                          Vectorize)
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten

from . import schedules

MASKED_LOGIT = -1e9


class View(nn.Module):
    sizes: tuple[int, ...]
    features: int
    dtype: jnp.dtype | None = None
    encoder: str = "linear"

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        view = obs["view"].astype(jnp.int32)
        planes = jax.nn.one_hot(
            view[..., 0], self.sizes[0], dtype=self.dtype or jnp.float32
        )
        if self.encoder == "cnn":
            planes = nn.Sequential(
                [
                    nn.Conv(16, (3, 3), strides=(2, 2), dtype=self.dtype),
                    nn.relu,
                    nn.Conv(32, (3, 3), dtype=self.dtype),
                    nn.relu,
                ]
            )(planes)
        state = nn.Dense(self.features, use_bias=False, dtype=self.dtype)(
            Flatten(start_dim=-3)(planes)
        )
        return {"state": state, "mask": obs["mask"]}


class Bypass(nn.Module):
    torso: nn.Module

    @nn.compact
    def __call__(self, carry, x, done, **kwargs):
        carry, state = self.torso(carry, x["state"], done, **kwargs)
        return carry, {"state": state, "mask": x["mask"]}

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.torso.initialize_carry(key, input_shape)


class Masked(nn.Module):
    num_actions: int
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.num_actions, use_bias=False, dtype=self.dtype)(
            x["state"]
        )
        return jnp.where(x["mask"], logits.astype(jnp.float32), MASKED_LOGIT)


def make(cfg):
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)

    num_actions = env.action_space().num_actions
    sizes = tuple(int(size) for size in env.observation_space()["view"].high + 1)
    hidden_dim = cfg.cell.features
    time_limit = env.time_limit()

    env = Vectorize(env, num_envs=cfg.environment.num_envs)
    env = Group(env, num_steps=time_limit)

    network = Network(
        feature_extractor=View(
            sizes=sizes,
            features=hidden_dim,
            dtype=dtype,
            encoder=(cfg.get("network") or {}).get("encoder", "linear"),
        ),
        torso=Bypass(torso=instantiate(cfg.stack)),
        head=ActorCritic(
            actor=Categorical(Masked(num_actions, dtype=dtype)),
            critic=nn.Sequential(
                [
                    operator.itemgetter("state"),
                    nn.Dense(1, use_bias=False, dtype=dtype),
                ]
            ),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg,
        batch_size=cfg.environment.num_envs * cfg.rollout.num_steps * env.num_agents,
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
            else optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
