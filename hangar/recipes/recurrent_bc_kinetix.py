from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import lox
import optax
from hydra.utils import instantiate
from omegaconf import OmegaConf

from boonta import datasets, environments
from boonta.algorithms.recurrent_bc import RecurrentBC
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize,
                                          Wrapper)
from boonta.networks import Block, Categorical, Network, ViTLayer
from boonta.utils.typing import Dtype

from . import schedules

MASKED = -1e9


class Entities(nn.Module):
    features: int
    num_layers: int
    num_heads: int
    values: tuple = ()
    prior: bool = False
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        dense = partial(nn.Dense, dtype=self.dtype, param_dtype=self.param_dtype)
        entity = obs["entities"]
        groups = [
            (entity.polygons, entity.polygon_mask),
            (entity.circles, entity.circle_mask),
            (entity.joints, entity.joint_mask),
            (entity.thrusters, entity.thruster_mask),
        ]
        tokens, valid = [], []
        for tag, (fields, mask) in enumerate(groups):
            embedded = dense(self.features)(fields)
            marker = jnp.zeros(embedded.shape[:-1] + (len(groups),), embedded.dtype)
            marker = marker.at[..., tag].set(1.0)
            tokens.append(dense(self.features)(
                jnp.concatenate([embedded, marker], axis=-1)
            ))
            valid.append(mask)
        tokens = jnp.concatenate(tokens, axis=-2)
        valid = jnp.concatenate(valid, axis=-1)

        lead = tokens.shape[:-2]
        entities, width = tokens.shape[-2], tokens.shape[-1]
        tokens = tokens.reshape(-1, entities, width)
        flat = valid.reshape(-1, entities)

        def attend(query_position, key_position, query_segment, key_segment, live):
            queries = jnp.take_along_axis(flat, query_position, axis=-1)
            keys = jnp.take_along_axis(flat, key_position, axis=-1) & live
            return (queries[:, :, None] & keys[:, None, :])[:, None]

        for index in range(self.num_layers):
            _, tokens = ViTLayer(
                features=self.features,
                num_heads=self.num_heads,
                attention_mask=attend,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"layers_{index}",
            )(None, tokens, jnp.zeros(tokens.shape[:2], bool))

        tokens = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(tokens)
        weight = flat[..., None].astype(jnp.float32)
        pooled = jnp.sum(tokens.astype(jnp.float32) * weight, axis=-2) / jnp.maximum(
            jnp.sum(weight, axis=-2), 1.0
        )
        pooled = pooled.reshape(*lead, width)

        if self.prior and action is not None and self.values:
            taken = jnp.asarray(action, jnp.int32)
            span = max(self.values)
            embedded = nn.Embed(
                len(self.values) * span,
                self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(taken + jnp.arange(len(self.values)) * span)
            pooled = pooled + jnp.sum(embedded.astype(jnp.float32), axis=-2)

        return {"features": pooled, "mask": obs["mask"]}


class Threaded(Block):
    inner: nn.Module

    @nn.compact
    def __call__(self, carry, x, done):
        carry, features = self.inner(carry, x["features"], done)
        return carry, {"features": features, "mask": x["mask"]}

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.inner.initialize_carry(key, input_shape)


class Motors(nn.Module):
    values: tuple
    depth: int = 5
    width: int = 128
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.Dense, dtype=self.dtype, param_dtype=self.param_dtype)
        hidden, active = x["features"], x["mask"]
        for _ in range(self.depth):
            hidden = nn.tanh(dense(self.width)(hidden))

        span = max(self.values)
        logits = dense(len(self.values) * span)(hidden).astype(jnp.float32)
        logits = logits.reshape(*hidden.shape[:-1], len(self.values), span)

        allowed = jnp.arange(span) < jnp.asarray(self.values)[:, None]
        logits = jnp.where(active[..., None], logits, 0.0)
        return jnp.where(allowed, logits, MASKED)


class Scored(Wrapper):
    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        succeeded = jnp.asarray(timestep.info["GoalR"], jnp.float32)
        lox.log(
            {
                "episode_statistics/success_rate": jnp.where(
                    timestep.done, succeeded, jnp.nan
                )
            }
        )
        return state, timestep

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)


class Dressed(Wrapper):
    def __init__(self, env, static):
        super().__init__(env)
        self.static = static

    def dress(self, state, timestep):
        from kinetix.data.bc_utils import get_valid_action_mask

        width = self.static.num_motor_bindings + self.static.num_thruster_bindings
        batch = jnp.shape(timestep.obs.polygon_mask)[0]
        action = jnp.zeros((batch, width), jnp.int32)
        mask = get_valid_action_mask(
            getattr(state, "unwrapped", state), self.static, action
        )
        return timestep.replace(obs={"entities": timestep.obs, "mask": mask})

    def init(self, key):
        state, timestep = self._env.init(key)
        return state, self.dress(state, timestep)

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        return state, self.dress(state, timestep)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)


def statics(cfg):
    from kinetix.environment import StaticEnvParams

    return StaticEnvParams(
        num_polygons=cfg.dataset.kwargs.num_polygons,
        num_circles=cfg.dataset.kwargs.num_circles,
        num_joints=cfg.dataset.kwargs.num_joints,
        num_thrusters=cfg.dataset.kwargs.num_thrusters,
        frame_skip=cfg.dataset.kwargs.frame_skip,
    )


def cardinality(static) -> tuple:
    return tuple([3] * static.num_motor_bindings + [2] * static.num_thruster_bindings)


def make(cfg):
    static = statics(cfg)
    dtype = (cfg.get("network") or {}).get("dtype")

    dataset = datasets.make(**cfg.dataset)

    spec = OmegaConf.to_container(cfg.environment, resolve=True)
    spec.setdefault("kwargs", {})["static_env_params"] = static
    if spec.pop("holdout_levels", False):
        source = spec.pop("levels_from", None)
        bank = dataset.levels()
        if source:
            elsewhere = OmegaConf.to_container(cfg.dataset, resolve=True)
            elsewhere["dataset_id"] = source
            bank = datasets.make(**elsewhere).levels()
        assert bank is not None, (
            "environment asks to evaluate on held-out dataset levels but the "
            "dataset reserved none. Set dataset.kwargs.val_shards > 0."
        )
        spec["env_id"] = None
        spec["kwargs"]["levels"] = bank
    env = environments.make(**spec)
    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=Entities(
            features=cfg.network.features,
            num_layers=cfg.network.num_layers,
            num_heads=cfg.network.num_heads,
            values=cardinality(static),
            prior=cfg.network.get("prior", False),
            dtype=dtype,
        ),
        torso=Threaded(instantiate(cfg.stack)),
        head=Categorical(
            Motors(
                values=cardinality(static),
                depth=cfg.network.actor_depth,
                width=cfg.network.actor_width,
                dtype=dtype,
            )
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=dataset.batch_size * dataset.length
    )

    algorithm = RecurrentBC(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env)
    env = Scored(env)
    env = Dressed(env, static)
    return {
        "algorithm": algorithm,
        "environment": env,
        "dataset": dataset,
        "pit": pit,
        "lap": lap,
    }
