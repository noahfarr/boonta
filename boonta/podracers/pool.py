"""BB-8's environment pool: on-policy training whose batch size changes during training.

A pool of `max(buckets)` environment states. One rollout-and-update step is compiled per bucket N. Each step gathers N
environments (round-robin, so every environment keeps moving), steps only those for T steps, updates on the N × T
batch, and scatters them back. Idle environments keep their state and are not stepped, and only active steps count.

Accounting. `train` scans over slots of `unit` environments × T steps (`batch_size`). Every slot adds `unit` environments
of credit, and an update of bucket N runs when the credit reaches N, which uses up the credit. The bucket can only
change right after an update, when the credit is zero, so every environment step taken is a step that was budgeted:
after k slots the algorithm has taken k · unit · T steps minus the credit still waiting (less than one batch).

Batch schedule (`schedule`):
- `fixed`: always `fixed_bucket`;
- `doubling`: `doubling_buckets` in equal fractions of `budget` steps;
- `noise`: target N_t = clip(κ · B_noise, min, max) with B_noise the algorithm's smoothed fresh-update noise scale of
  `noise_source` in environments; snapped to the nearest bucket in log scale, and only when the target crosses the
  geometric boundary next to the current bucket by a factor of `hysteresis`.

Step size (`step`), as a multiplier m on the optimizer's update (exactly the learning rate for Adam and SGD):
- `none`: m = 1;
- `lr`: m = m_max · B / (B + B_noise);  `lr_sqrt`: m = m_max · sqrt(B / (B + B_noise));
- `kl`: m adapts so the realized KL per update tracks KL_max · (B / (B + B_noise))²;  `kl_constant`: tracks KL_max.
The probe (the algorithm's `bb8.noise_scale_probe`) runs every `probe_every` updates; in between, the last smoothed
estimate is used. Before the first estimate B_noise is taken as B.

With a single bucket, no probe and `step: none`, `train` is Anakin's own.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
from flax import struct
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from boonta.algorithms import Algorithm
from boonta.utils import Key, PyTree, sharded
from boonta.utils.typing import Environment

from .anakin import Anakin, AnakinConfig, AnakinState
from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class PoolConfig(AnakinConfig):
    buckets: tuple = struct.field(pytree_node=False, default=(256, 512, 1024, 2048, 4096, 8192))
    schedule: str = struct.field(pytree_node=False, default="fixed")
    fixed_bucket: int = struct.field(pytree_node=False, default=8192)
    doubling_buckets: tuple = struct.field(pytree_node=False, default=(256, 512, 1024, 2048, 4096))
    budget: int = struct.field(pytree_node=False, default=0)
    kappa: float = struct.field(pytree_node=False, default=1.0)
    hysteresis: float = struct.field(pytree_node=False, default=1.5)
    initial_bucket: int = struct.field(pytree_node=False, default=256)
    probe_every: int = struct.field(pytree_node=False, default=8)
    noise_source: str = struct.field(pytree_node=False, default="actor")
    step: str = struct.field(pytree_node=False, default="none")
    multiplier_max: float = struct.field(pytree_node=False, default=1.0)
    kl_max: float = struct.field(pytree_node=False, default=0.01)
    minibatch_envs: int = struct.field(pytree_node=False, default=512)
    running_normalization_below: int = struct.field(pytree_node=False, default=0)
    noise_scale_cap: float = struct.field(pytree_node=False, default=32768.0)


@struct.dataclass(frozen=True)
class PoolState(AnakinState):
    credit: jax.Array = 0
    offset: jax.Array = 0
    bucket: jax.Array = 0
    updates: jax.Array = 0
    multiplier: jax.Array = 1.0


def with_num_envs(environment, num_envs):
    """A copy of the wrapper chain whose vectorizing wrapper steps `num_envs` environments."""
    # copy.copy would recurse through Wrapper.__getattr__ before `_env` exists on the copy.
    copied = object.__new__(type(environment))
    copied.__dict__.update(vars(environment))
    environment = copied
    if "num_envs" in vars(environment):
        environment.num_envs = num_envs
    inner = vars(environment).get("_env")
    if inner is not None:
        environment._env = with_num_envs(inner, num_envs)
    return environment


@dataclass
class Pool(Anakin):
    """`config` is a PoolConfig."""

    def __post_init__(self):
        super().__post_init__()
        cfg = self.config
        self.size = max(cfg.buckets)
        self.unit = min(cfg.buckets)
        assert self.environment.num_envs == self.size, "environment.num_envs must be the largest bucket"
        assert all(b % self.unit == 0 for b in cfg.buckets)
        self.plain = cfg.schedule == "fixed" and cfg.fixed_bucket == self.size and cfg.step == "none" and not self.probing
        self.optimizer = self.algorithm.optimizer

    @property
    def probing(self) -> bool:
        return self.config.schedule == "noise" or self.config.step in ("lr", "lr_sqrt", "kl")

    @property
    def batch_size(self) -> int:
        if self.plain:
            return super().batch_size
        return self.unit * self.config.num_steps * self.environment.num_agents

    def variant(self, bucket: int, probe: bool, multiplier) -> Algorithm:
        """The algorithm for a batch of `bucket` environments: fixed minibatch size, optional probe, scaled step."""
        cfg = self.algorithm.cfg
        changes = {"num_minibatches": max(1, bucket // self.config.minibatch_envs)}
        if hasattr(cfg, "bb8"):
            changes["bb8"] = dataclasses.replace(
                cfg.bb8,
                noise_scale_probe=probe,
                running_advantage_normalization=bucket < self.config.running_normalization_below,
            )
        return dataclasses.replace(
            self.algorithm,
            cfg=cfg.replace(**changes),
            optimizer=optax.chain(self.optimizer, optax.scale(multiplier)),
        )

    def init(self, key: Key) -> PoolState:
        cfg = self.algorithm.cfg
        if hasattr(cfg, "bb8") and not self.plain:
            base = dataclasses.replace(
                cfg.bb8,
                noise_scale_probe=self.probing,
                running_advantage_normalization=self.config.running_normalization_below > 0,
            )
            self.algorithm = dataclasses.replace(
                self.algorithm, cfg=cfg.replace(bb8=base), optimizer=optax.chain(self.optimizer, optax.scale(1.0))
            )
        state = super().init(key)
        if hasattr(cfg, "bb8") and not self.plain:
            self.algorithm = dataclasses.replace(self.algorithm, cfg=cfg, optimizer=self.optimizer)
        initial = self.config.fixed_bucket if self.config.schedule == "fixed" else (
            self.config.doubling_buckets[0] if self.config.schedule == "doubling" else self.config.initial_bucket)
        return PoolState(
            **{f.name: getattr(state, f.name) for f in dataclasses.fields(AnakinState)},
            credit=jnp.zeros((), jnp.int32),
            offset=jnp.zeros((), jnp.int32),
            bucket=jnp.asarray(self.config.buckets.index(initial), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            multiplier=jnp.asarray(self.step_target(initial, jnp.nan) if self.config.step in ("lr", "lr_sqrt") else 1.0,
                                   jnp.float32),
        )

    def noise_scale(self, algorithm_state, name):
        statistics = algorithm_state.bb8
        if statistics is None or "noise" not in statistics or name not in statistics["noise"]:
            return jnp.nan
        noise, signal = statistics["noise"][name]["fresh"], statistics["signal"][name]["fresh"]
        scale = jnp.where(signal > 0, noise / signal, self.config.noise_scale_cap)
        return jnp.where(statistics["count"] > 0, jnp.clip(scale, 0.0, self.config.noise_scale_cap), jnp.nan)

    def step_target(self, bucket, scale):
        cfg = self.config
        ratio = bucket / (bucket + jnp.where(jnp.isnan(scale), bucket, scale))
        return {
            "none": 1.0,
            "lr": cfg.multiplier_max * ratio,
            "lr_sqrt": cfg.multiplier_max * jnp.sqrt(ratio),
            "kl": cfg.kl_max * ratio**2,
            "kl_constant": cfg.kl_max,
        }[cfg.step]

    def next_bucket(self, state: PoolState, scale) -> jax.Array:
        cfg = self.config
        buckets = jnp.asarray(cfg.buckets, jnp.float32)
        if cfg.schedule == "fixed":
            return state.bucket
        if cfg.schedule == "doubling":
            progress = state.algorithm_state.step.astype(jnp.float32) / cfg.budget
            stage = jnp.clip(jnp.floor(progress * len(cfg.doubling_buckets)), 0, len(cfg.doubling_buckets) - 1)
            chosen = jnp.asarray(cfg.doubling_buckets, jnp.float32)[stage.astype(jnp.int32)]
            return jnp.argmin(jnp.abs(buckets - chosen)).astype(jnp.int32)
        target = jnp.clip(cfg.kappa * jnp.where(jnp.isnan(scale), buckets[state.bucket], scale), buckets[0], buckets[-1])
        snapped = jnp.argmin(jnp.abs(jnp.log(buckets) - jnp.log(target))).astype(jnp.int32)
        current = buckets[state.bucket]
        moved = (target > current * np.sqrt(2) * cfg.hysteresis) | (target < current / np.sqrt(2) / cfg.hysteresis)
        return jnp.where(moved, snapped, state.bucket)

    def bucket_step(self, index: int, state: PoolState, key: Key) -> PoolState:
        """One rollout and update of bucket `self.config.buckets[index]`; logs scalars only."""
        cfg = self.config
        bucket = cfg.buckets[index]
        rollout_key, update_key = jax.random.split(key)
        indices = (state.offset + jnp.arange(bucket)) % self.size
        take = partial(jax.tree.map, lambda leaf: jnp.take(leaf, indices, axis=0))
        sub = AnakinState(
            timestep=take(state.timestep),
            environment_state=take(state.environment_state),
            carry=take(state.carry),
            algorithm_state=state.algorithm_state,
            pit_state=state.pit_state,
            lap_state=state.lap_state,
        )
        used = state.multiplier
        racer = Anakin(AnakinConfig(num_steps=cfg.num_steps, mesh=cfg.mesh), self.variant(bucket, False, used),
                       with_num_envs(self.environment, bucket), self.pit, self.lap)
        (sub, transitions), rollout_logs = lox.spool(
            lambda s, keys: jax.lax.scan(partial(Anakin.rollout, racer, temperature=1.0), s, keys)
        )(sub, jax.random.split(rollout_key, cfg.num_steps))
        before = sub.algorithm_state

        def update(probe):
            algorithm = self.variant(bucket, probe, state.multiplier)
            out, logs = lox.spool(algorithm.update)(before, update_key, transitions)
            return out, dict(logs)

        if self.probing:
            probed_shape = jax.eval_shape(lambda: update(True)[1])
            def without_probe():
                out, logs = update(False)
                filled = {k: jnp.full(v.shape, jnp.nan, v.dtype) for k, v in probed_shape.items()}
                return out, {k: logs.get(k, filled[k]) for k in sorted(filled)}
            after, update_logs = jax.lax.cond(state.updates % cfg.probe_every == 0, lambda: (lambda o, l: (o, {k: l[k] for k in sorted(l)}))(*update(True)), without_probe)
        else:
            after, update_logs = update(False)

        scale = self.noise_scale(after, cfg.noise_source)
        flat = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), transitions.first.obs)
        output_before = self.algorithm.network.apply(before.params, flat, temperature=1.0)
        output_after = self.algorithm.network.apply(after.params, flat, temperature=1.0)
        if isinstance(output_before, tuple):
            old = jax.nn.log_softmax(output_before[0].logits, -1)
            new = jax.nn.log_softmax(output_after[0].logits, -1)
            kl = jnp.mean(jnp.sum(jnp.exp(old) * (old - new), -1))
        else:
            kl = jnp.nan
        moved = jax.tree.map(lambda a, b: a - b, after.params["params"], before.params["params"])
        update_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(moved)))

        put = lambda full, part: jax.tree.map(lambda f, p: f.at[indices].set(p), full, part)
        state = state.replace(
            timestep=put(state.timestep, sub.timestep),
            environment_state=put(state.environment_state, sub.environment_state),
            carry=put(state.carry, sub.carry),
            algorithm_state=after,
            lap_state=sub.lap_state,
            credit=state.credit - bucket,
            offset=(state.offset + bucket) % self.size,
            updates=state.updates + 1,
        )
        next_bucket = self.next_bucket(state, scale)
        if cfg.step in ("kl", "kl_constant"):
            # The realized KL of this update against this update's target; the multiplier carries over.
            target = self.step_target(bucket, scale)
            correction = jnp.clip(jnp.sqrt(target / jnp.maximum(kl, 1e-12)), 0.5, 2.0)
            multiplier = jnp.clip(used * correction, 1.0 / 256, 16.0)
        else:
            # The learning-rate rule for the next update's batch.
            target = self.step_target(jnp.asarray(cfg.buckets, jnp.float32)[next_bucket], scale)
            multiplier = jnp.asarray(target, jnp.float32)
        state = state.replace(bucket=next_bucket, multiplier=multiplier)

        logs = {k: jnp.nanmean(v) for k, v in update_logs.items()}
        for name, value in rollout_logs.items():
            logs[name] = jnp.nanmean(value)
            logs[f"{name}_sum"] = jnp.nansum(value)
            logs[f"{name}_count"] = jnp.sum(jnp.isfinite(value)).astype(jnp.float32)
        logs |= {
            "pool/bucket": jnp.asarray(bucket, jnp.float32),
            "pool/noise_scale_envs": scale,
            "pool/noise_scale_critic_envs": self.noise_scale(after, "critic"),
            "pool/step_target": jnp.asarray(target, jnp.float32),
            "pool/multiplier": used,
            "pool/kl": kl,
            "pool/update_norm": update_norm,
            "pool/env_steps": after.step.astype(jnp.float32),
        }
        lox.log(logs)
        return state

    def train(self, state: PoolState, key: Key, num_updates: int) -> PyTree:
        if self.plain:
            return Anakin.train(self, state, key, num_updates)

        branches = [partial(self.bucket_step, index) for index in range(len(self.config.buckets))]

        def slot(state: PoolState, key):
            state = state.replace(credit=state.credit + self.unit)
            due = state.credit >= jnp.asarray(self.config.buckets, jnp.int32)[state.bucket]
            state = jax.lax.cond(due, lambda s: jax.lax.switch(s.bucket, branches, s, key), lambda s: s, state)
            return state, None

        state, _ = jax.lax.scan(slot, state, jax.random.split(key, num_updates))
        return state


def make(
    config: PoolConfig,
    algorithm: Algorithm,
    environment: Environment,
    pit: Pit | None = None,
    lap: Lap | None = None,
) -> Pool:
    podracer = Pool(config, algorithm, environment, pit, lap)

    initialize, close = podracer.init, podracer.close
    train = lox.spool(lox.strip(podracer.train, tags=["evaluation"]), unify=True)
    evaluate = lox.spool(lox.strip(Anakin.evaluate.__get__(podracer), tags=["training"]))

    replicated = NamedSharding(config.mesh, P())
    axes = sharded(jax.eval_shape(initialize, jax.random.key(0)))
    shardings = jax.tree.map(
        lambda axis: NamedSharding(config.mesh, P(axis)) if axis else replicated, axes
    )
    podracer.init = jax.jit(initialize, out_shardings=shardings)
    podracer.train = jax.jit(
        train,
        static_argnums=(2,),
        donate_argnames=("state",),
        in_shardings=(shardings, replicated),
        out_shardings=(shardings, replicated),
    )
    podracer.evaluate = jax.jit(evaluate, static_argnums=(2,), in_shardings=(shardings, replicated),
                                out_shardings=(shardings, replicated))
    podracer.close = jax.jit(close, in_shardings=(shardings,))
    return podracer
