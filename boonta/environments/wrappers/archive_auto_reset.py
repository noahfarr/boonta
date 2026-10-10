from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.utils import Array, Key, PyTree, broadcast, remove_batch_axis
from boonta.utils.table import claim

from .auto_reset import AutoReset
from .time_limit import TimeLimitState
from .wrapper import WrapperState


@struct.dataclass
class CellState:
    snapshot: PyTree
    time: Array
    rho: Array
    eligible: Array


def admits(state: CellState, rho: Array, eviction_mask: Array, eligible: Array) -> Array:
    upgrade = eligible & ~state.eligible
    downgrade = state.eligible & ~eligible
    return eviction_mask | upgrade | (~downgrade & (rho | ~state.rho))


@struct.dataclass
class ArchiveState:
    keys: Array
    mask: Array
    cell_states: CellState
    clock: Array


def noeviction(state: ArchiveState) -> Array:
    return jnp.full(jnp.shape(state.mask), jnp.inf)


def lru(state: ArchiveState) -> Array:
    return state.cell_states.time.astype(jnp.float32)


def anywhere(env) -> Callable[[PyTree], Array]:
    return lambda env_state: jnp.bool_(True)


@dataclass
class Archive:
    archive_size: int
    cell_fn: Callable[[PyTree], Array]
    eviction_fn: Callable[[ArchiveState], Array] = noeviction
    num_probes: int = 8

    def init(self, env_state: PyTree) -> ArchiveState:
        _, *key_shape = jax.eval_shape(self.cell_fn, env_state).shape
        snapshot = jax.eval_shape(lambda tree: jax.tree.map(remove_batch_axis, tree), env_state)
        cell_state = CellState(
            snapshot=snapshot,
            time=jax.ShapeDtypeStruct((), jnp.int32),
            rho=jax.ShapeDtypeStruct((), jnp.bool),
            eligible=jax.ShapeDtypeStruct((), jnp.bool),
        )
        return ArchiveState(
            keys=jnp.zeros((self.archive_size, *key_shape), jnp.int32),
            mask=jnp.zeros(self.archive_size, jnp.bool),
            cell_states=jax.tree.map(
                lambda leaf: jnp.zeros((self.archive_size, *leaf.shape), leaf.dtype),
                cell_state,
            ),
            clock=jnp.zeros((), jnp.int32),
        )

    def at(self, state: ArchiveState, slot: Array) -> CellState:
        return jax.tree.map(
            lambda leaf: leaf[jnp.clip(slot, 0)], state.cell_states.replace(snapshot=None)
        )

    def put(self, state: ArchiveState, slot: Array, cell_state: CellState) -> ArchiveState:
        slot = jnp.where(slot >= 0, slot, self.archive_size)
        cell_states = jax.tree.map(
            lambda held, new: held.at[slot].set(new, mode="drop"), state.cell_states, cell_state
        )
        return state.replace(cell_states=cell_states)

    def add(
        self,
        state: ArchiveState,
        env_state: PyTree,
        done: Array,
        rho: Array | None = None,
        eligible: Array | None = None,
    ) -> tuple[ArchiveState, Array, Array, Array]:
        rho = jnp.ones_like(done) if rho is None else rho
        eligible = jnp.ones_like(done) if eligible is None else eligible
        cell_keys = self.cell_fn(env_state)
        archive_keys, archive_mask, claimed_slots, opened_mask = claim(
            state.keys,
            state.mask,
            cell_keys,
            done,
            self.num_probes,
            self.eviction_fn(state),
        )
        clipped_slots = jnp.clip(claimed_slots, 0)
        eviction_mask = (
            opened_mask
            & state.mask[clipped_slots]
            & jnp.any(state.keys[clipped_slots] != archive_keys[clipped_slots], axis=-1)
        )
        admission_mask = (claimed_slots >= 0) & admits(
            self.at(state, clipped_slots), rho, eviction_mask, eligible
        )
        order = jnp.arange(jnp.shape(done)[0], dtype=jnp.float32)
        write_priorities = (
            2.0 * eligible.astype(jnp.float32)
            + rho.astype(jnp.float32)
            + 0.5 * (1.0 - order / (order.shape[0] + 1.0))
        )
        highest_priority = (
            jnp.full(self.archive_size, -1.0)
            .at[jnp.where(admission_mask, clipped_slots, self.archive_size)]
            .max(write_priorities, mode="drop")
        )
        write_mask = admission_mask & (write_priorities == highest_priority[clipped_slots])
        cell_state = CellState(
            snapshot=env_state,
            time=jnp.broadcast_to(state.clock, jnp.shape(done)),
            rho=rho,
            eligible=eligible,
        )
        state = self.put(
            state.replace(keys=archive_keys, mask=archive_mask, clock=state.clock + 1),
            jnp.where(write_mask, clipped_slots, -1),
            cell_state,
        )
        evicted_slots = jnp.where(eviction_mask, claimed_slots, -1)
        return state, claimed_slots, opened_mask, evicted_slots

    def take(self, state: ArchiveState, slot: Array) -> PyTree:
        return jax.tree.map(lambda leaf: leaf[jnp.clip(slot, 0)], state.cell_states.snapshot)

    def eligible(self, state: ArchiveState) -> Array:
        return state.mask & state.cell_states.eligible


def pick(key: Key, logits: Array, shape) -> Array:
    highest = jnp.max(logits)
    weights = jnp.where(jnp.isfinite(logits), jnp.exp(logits - highest), 0.0)
    cumulative = jnp.cumsum(weights)
    draws = jax.random.uniform(key, shape, cumulative.dtype) * cumulative[-1]
    picked = jnp.searchsorted(cumulative, draws, side="right")
    return jnp.clip(picked, 0, jnp.shape(logits)[0] - 1).astype(jnp.int32)


def share(k: float, num_envs: int) -> int:
    assert k >= 1.0, f"k must be at least 1, got {k}"
    return int(round((1.0 - 1.0 / k) * num_envs))


def widen(cell_fn: Callable[[PyTree], Array]) -> Callable[[PyTree], Array]:
    def widened(env_state):
        tag = cell_fn(env_state)
        return tag if jnp.ndim(tag) > 1 else tag[..., None]

    return widened


class Selector:
    def init(self, archive_size: int, num_actions: int) -> PyTree:
        return ()

    def update(self, state: PyTree, key: Key, transitions: PyTree) -> PyTree:
        return state

    def select(
        self, state: PyTree, wrapper, archive_auto_reset_state, key: Key
    ) -> tuple[PyTree, Array]:
        raise NotImplementedError


@struct.dataclass
class ArchiveAutoResetState(WrapperState):
    archive_state: ArchiveState = struct.field(metadata={"axis": None})
    selector_state: PyTree = struct.field(metadata={"axis": None})
    assigned_slots: Array
    due_mask: Array
    current_slots: Array
    rho: Array
    age: Array
    earned: Array
    graded: Array


def reset_time_limit(state: PyTree) -> PyTree:
    if isinstance(state, TimeLimitState):
        state = state.replace(time=jnp.zeros_like(state.time))
    if isinstance(state, WrapperState):
        state = state.replace(env_state=reset_time_limit(state.env_state))
    return state


def locate(state: PyTree) -> ArchiveAutoResetState:
    while not isinstance(state, ArchiveAutoResetState):
        state = state.env_state
    return state


def plant(state: PyTree, archive_auto_reset_state: ArchiveAutoResetState) -> PyTree:
    if isinstance(state, ArchiveAutoResetState):
        return archive_auto_reset_state
    return state.replace(env_state=plant(state.env_state, archive_auto_reset_state))


def unwrap(environment):
    env = environment
    while not isinstance(env, ArchiveAutoReset):
        env = getattr(env, "_env", None)
        if env is None:
            raise ValueError(
                "restart curricula drive an ArchiveAutoReset, and the environment has none"
            )
    return env


class ArchiveAutoReset(AutoReset):
    def __init__(
        self,
        env,
        num_envs: int,
        cell_fn: Callable,
        selector: Selector,
        capacity: int = 16384,
        num_probes: int = 8,
        eviction_fn: Callable[[ArchiveState], Array] = noeviction,
        eligibility_fn: Callable = anywhere,
        gamma: float = 0.99,
        refresh_fn: Callable[[PyTree, Key], PyTree] = lambda env_state, key: env_state,
    ):
        super().__init__(env, num_envs=num_envs)
        self.selector = selector
        self.actions = int(getattr(env.action_space(), "num_actions", 1))
        self.archive = Archive(
            archive_size=capacity,
            cell_fn=widen(cell_fn(env=env)),
            eviction_fn=eviction_fn,
            num_probes=num_probes,
        )
        self.eligibility_fn = eligibility_fn(env=env)
        self.gamma = gamma
        self.refresh_fn = refresh_fn

    def block(self, num_placed: int) -> Array:
        return jnp.arange(self.num_envs) >= self.num_envs - num_placed

    def occupied(self, state, num_placed: int) -> Array:
        archive_size = self.archive.archive_size
        held_mask = ~self.block(num_placed) & (state.current_slots >= 0)
        return (
            jnp.zeros(archive_size)
            .at[jnp.where(held_mask, state.current_slots, archive_size)]
            .add(1.0, mode="drop")
        )

    def eligible(self, env_state: PyTree) -> Array:
        eligible = jnp.asarray(self.eligibility_fn(env_state))
        if jnp.ndim(eligible) == 0:
            return jnp.broadcast_to(eligible, (self.num_envs,))
        return jnp.reshape(eligible, (self.num_envs, -1)).all(axis=-1)

    def init(self, key: Key):
        env_state, timestep = super().init(key)
        archive_state = self.archive.init(env_state)
        archive_state, claimed_slots, _, evicted_slots = self.archive.add(
            archive_state,
            reset_time_limit(env_state),
            jnp.ones(self.num_envs, bool),
        )
        state = ArchiveAutoResetState(
            env_state=env_state,
            archive_state=archive_state,
            selector_state=self.selector.init(self.archive.archive_size, self.actions),
            assigned_slots=jnp.full(self.num_envs, -1, jnp.int32),
            due_mask=jnp.zeros(self.num_envs, bool),
            current_slots=claimed_slots,
            rho=jnp.ones(self.num_envs, bool),
            age=jnp.zeros(self.num_envs, jnp.int32),
            earned=jnp.zeros(self.num_envs),
            graded=jnp.zeros(self.num_envs),
        )
        return state, timestep.replace(
            info=self.mark(
                state, timestep, claimed_slots, jnp.ones(self.num_envs, bool), evicted_slots
            ),
        )

    def step(self, key: Key, state, action):
        step_key, reset_key, seed_key = jax.random.split(key, 3)
        rho = state.rho
        age = state.age + 1
        env_state, timestep = super().step(step_key, state.env_state, action)
        state = state.replace(env_state=env_state, age=age)
        episode_mask = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        truncation_mask = state.due_mask & ~episode_mask
        timestep = timestep.replace(
            truncated=timestep.truncated | broadcast(truncation_mask, timestep.truncated)
        )
        done = episode_mask | truncation_mask
        restart_mask = done & (state.assigned_slots >= 0)
        env_states, _ = jax.vmap(self._env.init)(
            jax.random.split(reset_key, self.num_envs)
        )

        def load_snapshots():
            archived_states = self.archive.take(state.archive_state, state.assigned_slots)
            archived_states = self.refresh_fn(archived_states, seed_key)
            return jax.tree.map(
                lambda archived, reset: jnp.where(
                    broadcast(restart_mask, reset), archived, reset
                ),
                archived_states,
                env_states,
            )

        env_states = jax.lax.cond(
            jnp.any(restart_mask), load_snapshots, lambda: env_states
        )
        state = state.replace(
            env_state=jax.tree.map(
                lambda reset, live: jnp.where(broadcast(done, live), reset, live),
                env_states,
                state.env_state,
            ),
            rho=jnp.where(done, ~restart_mask, state.rho),
            age=jnp.where(done, 0, state.age),
            due_mask=jnp.zeros_like(state.due_mask),
        )
        timestep = timestep.replace(
            obs=jax.tree.map(
                lambda spawned, live: jnp.where(broadcast(done, live), spawned, live),
                self.observe(state),
                timestep.obs,
            )
        )
        archive_state, claimed_slots, opened_mask, evicted_slots = self.archive.add(
            state.archive_state,
            reset_time_limit(state.env_state),
            ~restart_mask,
            state.rho,
            done | self.eligible(state.env_state),
        )
        current_slots = jnp.where(restart_mask, state.assigned_slots, claimed_slots)
        decay = self.gamma ** (age - 1).astype(jnp.float32)
        reward = timestep.reward.reshape(self.num_envs, -1).sum(axis=-1)
        earned = state.earned + reward
        graded = state.graded + decay * reward

        lox.log(
            {
                "archive/miss": jnp.mean((current_slots < 0).astype(jnp.float32)),
                "archive/discovery_rate": jnp.sum(opened_mask, dtype=jnp.float32),
                "archive/rho": jnp.sum(archive_state.cell_states.rho, dtype=jnp.float32)
                / jnp.maximum(jnp.sum(archive_state.mask, dtype=jnp.float32), 1.0),
                "archive/num_evictions": jnp.sum(evicted_slots >= 0, dtype=jnp.float32),
            }
        )
        self.recount(earned, graded, age, rho, episode_mask)

        return (
            state.replace(
                archive_state=archive_state,
                current_slots=current_slots,
                earned=jnp.where(done, 0.0, earned),
                graded=jnp.where(done, 0.0, graded),
            ),
            timestep.replace(
                info=self.mark(
                    state,
                    timestep,
                    current_slots,
                    done & ~restart_mask,
                    evicted_slots,
                )
            ),
        )

    def mark(self, state, timestep, current_slots, start_mask, evicted_slots) -> dict:
        info = timestep.info if isinstance(timestep.info, dict) else {}
        return {
            **info,
            "cell": current_slots,
            "rho": state.rho,
            "start": start_mask,
            "evicted": evicted_slots,
        }

    def recount(self, earned, graded, age, rho, done) -> None:
        length = age.astype(jnp.float32)
        lox.log(
            {
                "mu/episode_return": jnp.where(done, earned, jnp.nan),
                "mu/discounted_episode_return": jnp.where(done, graded, jnp.nan),
                "mu/episode_length": jnp.where(done, length, jnp.nan),
                "rho/episode_return": jnp.where(done & rho, earned, jnp.nan),
                "rho/discounted_episode_return": jnp.where(done & rho, graded, jnp.nan),
                "rho/episode_length": jnp.where(done & rho, length, jnp.nan),
            }
        )

    def assign(self, state, key: Key, transitions: PyTree):
        update_key, select_key = jax.random.split(key)
        if transitions is not None:
            state = state.replace(
                selector_state=self.selector.update(state.selector_state, update_key, transitions)
            )
        selector_state, selected_slots = self.selector.select(
            state.selector_state, self, state, select_key
        )
        num_placed, *_ = jnp.shape(selected_slots)
        block_mask = self.block(num_placed)
        assigned_slots = (
            jnp.full(self.num_envs, -1, jnp.int32)
            .at[self.num_envs - num_placed :]
            .set(selected_slots)
        )
        lox.log(
            {
                "archive/placed": jnp.sum(block_mask, dtype=jnp.float32),
                "archive/num_cells": jnp.sum(state.archive_state.mask, dtype=jnp.float32),
                "archive/num_eligible": jnp.sum(
                    self.archive.eligible(state.archive_state), dtype=jnp.float32
                ),
            }
        )
        return state.replace(
            selector_state=selector_state,
            assigned_slots=jnp.where(block_mask, assigned_slots, -1),
            due_mask=block_mask,
        )

    def update(self, state, key: Key, **kwargs):
        return state.replace(env_state=super().update(state.env_state, key, **kwargs))

    def action_mask(self, state) -> Array | None:
        return super().action_mask(state.env_state)

    def observe(self, state) -> PyTree:
        return super().observe(state.env_state)
