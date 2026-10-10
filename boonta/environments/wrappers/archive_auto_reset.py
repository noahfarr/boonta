from collections.abc import Callable
from dataclasses import dataclass, replace

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
class Snapshot:
    env_state: PyTree
    carry: PyTree


@struct.dataclass
class CellState:
    snapshots: Snapshot
    head: Array
    mask: Array
    identities: Array
    time: Array
    rho: Array
    eligible: Array


@dataclass
class Cell:
    size: int = 1

    def init(self, snapshot: Snapshot) -> CellState:
        return CellState(
            snapshots=jax.tree.map(
                lambda leaf: jnp.zeros((self.size, *leaf.shape), leaf.dtype), snapshot
            ),
            head=jnp.zeros((), jnp.int32),
            mask=jnp.zeros(self.size, bool),
            identities=jnp.full(self.size, -1, jnp.int32),
            time=jnp.zeros((), jnp.int32),
            rho=jnp.zeros((), bool),
            eligible=jnp.zeros((), bool),
        )

    def holds(self, state: CellState, identity: Array) -> Array:
        return jnp.any(state.mask & (state.identities == identity[..., None]), axis=-1)

    def admits(
        self, state: CellState, rho: Array, dirty: Array, eligible: Array, known: Array
    ) -> Array:
        upgrade = eligible & ~state.eligible
        downgrade = state.eligible & ~eligible
        return dirty | (~known & (upgrade | (~downgrade & (rho | ~state.rho))))

    def add(
        self,
        state: CellState,
        env_state: PyTree,
        rho: Array,
        clock: Array,
        dirty: Array,
        eligible: Array,
        identity: Array,
    ) -> CellState:
        head = jnp.where(dirty, 0, state.head)
        mask = jnp.where(dirty, jnp.zeros_like(state.mask), state.mask)
        identities = jnp.where(dirty, jnp.full_like(state.identities, -1), state.identities)
        snapshots = (
            None
            if state.snapshots is None
            else state.snapshots.replace(
                env_state=jax.tree.map(
                    lambda held, new: held.at[head].set(new),
                    state.snapshots.env_state,
                    env_state,
                )
            )
        )
        return state.replace(
            snapshots=snapshots,
            head=(head + 1) % self.size,
            mask=mask.at[head].set(True),
            identities=identities.at[head].set(identity),
            time=clock,
            rho=rho,
            eligible=eligible,
        )

    def sample(self, state: CellState, key: Key) -> Array:
        logits = jnp.where(state.mask, 0.0, -jnp.inf)
        return jnp.where(jnp.any(state.mask), jax.random.categorical(key, logits), 0)


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


@dataclass
class Archive:
    archive_size: int
    cell_fn: Callable[[PyTree], Array]
    eviction_fn: Callable[[ArchiveState], Array] = noeviction
    carry_shape: PyTree = None
    cell_size: int = 1
    num_probes: int = 8
    identity_fn: Callable[[PyTree], Array] | None = None

    def __post_init__(self):
        self.cell = Cell(size=self.cell_size)

    def init(self, env_state: PyTree) -> ArchiveState:
        _, *key_shape = jax.eval_shape(self.cell_fn, env_state).shape
        snapshot = jax.eval_shape(
            lambda tree: jax.tree.map(remove_batch_axis, tree),
            Snapshot(env_state=env_state, carry=self.carry_shape),
        )
        cell_state = jax.eval_shape(self.cell.init, snapshot)
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
            lambda leaf: leaf[jnp.clip(slot, 0)], state.cell_states.replace(snapshots=None)
        )

    def put(self, state: ArchiveState, slot: Array, cell_state: CellState) -> ArchiveState:
        slot = jnp.where(slot >= 0, slot, self.archive_size)
        held = jax.tree.map(
            lambda held, new: held.at[slot].set(new, mode="drop"),
            state.cell_states.replace(snapshots=None),
            cell_state.replace(snapshots=None),
        )
        return state.replace(cell_states=held.replace(snapshots=state.cell_states.snapshots))

    def write(
        self, state: ArchiveState, slot: Array, position: Array, field: str, value: PyTree
    ) -> ArchiveState:
        row = jnp.where((slot >= 0) & (position >= 0), slot, self.archive_size)
        column = jnp.clip(position, 0)
        snapshots = state.cell_states.snapshots
        stored = jax.tree.map(
            lambda held, new: held.at[row, column].set(new, mode="drop"),
            getattr(snapshots, field),
            value,
        )
        return state.replace(
            cell_states=state.cell_states.replace(
                snapshots=snapshots.replace(**{field: stored})
            )
        )

    def add(
        self,
        state: ArchiveState,
        env_state: PyTree,
        done: Array,
        rho: Array | None = None,
        eligible: Array | None = None,
    ) -> tuple[ArchiveState, Array, Array, Array, Array]:
        rho = jnp.ones_like(done) if rho is None else rho
        eligible = jnp.ones_like(done) if eligible is None else eligible
        key = self.cell_fn(env_state)
        keys, mask, index, opened = claim(
            state.keys,
            state.mask,
            key,
            done,
            self.num_probes,
            self.eviction_fn(state),
        )
        slot = jnp.clip(index, 0)
        dirty = opened & state.mask[slot] & jnp.any(state.keys[slot] != keys[slot], axis=-1)
        cell_state = self.at(state, slot)
        if self.identity_fn is None:
            identity = jnp.full(jnp.shape(done), -1, jnp.int32)
            known = jnp.zeros_like(dirty)
        else:
            identity = self.identity_fn(env_state).astype(jnp.int32)
            known = self.cell.holds(cell_state, identity)
        writes = (index >= 0) & self.cell.admits(cell_state, rho, dirty, eligible, known)
        order = jnp.arange(jnp.shape(done)[0], dtype=jnp.float32)
        priority = (
            2.0 * eligible.astype(jnp.float32)
            + rho.astype(jnp.float32)
            + 0.5 * (1.0 - order / (order.shape[0] + 1.0))
        )
        writer = (
            jnp.full(self.archive_size, -1.0)
            .at[jnp.where(writes, slot, self.archive_size)]
            .max(priority, mode="drop")
        )
        writes = writes & (priority == writer[slot])
        added = jax.vmap(self.cell.add)(
            cell_state,
            env_state,
            rho,
            jnp.broadcast_to(state.clock, jnp.shape(done)),
            dirty,
            eligible,
            identity,
        )
        cell_state = jax.tree.map(
            lambda new, old: jnp.where(broadcast(writes, new), new, old), added, cell_state
        )
        position = jnp.where(writes, jnp.where(dirty, 0, self.at(state, slot).head), -1)
        state = self.put(
            state.replace(keys=keys, mask=mask, clock=state.clock + 1),
            jnp.where(writes, slot, -1),
            cell_state,
        )
        state = self.write(state, slot, position, "env_state", env_state)
        return state, index, position, opened, jnp.where(dirty, index, -1)

    def stow(self, state: ArchiveState, slot: Array, position: Array, carry: PyTree) -> ArchiveState:
        if self.carry_shape is None:
            return state
        return self.write(state, slot, position, "carry", carry)

    def locate(self, state: ArchiveState, index: Array, key: Key) -> Array:
        subkeys = jax.random.split(key, jnp.shape(index)[0])
        position = jax.vmap(self.cell.sample)(self.at(state, index), subkeys)
        return jnp.where(index >= 0, position, -1)

    def take(self, state: ArchiveState, slot: Array, position: Array) -> Snapshot:
        return jax.tree.map(
            lambda leaf: leaf[jnp.clip(slot, 0), jnp.clip(position, 0)],
            state.cell_states.snapshots,
        )

    def eligible(self, state: ArchiveState) -> Array:
        return state.mask & state.cell_states.eligible


def pick(key: Key, logits: Array, shape) -> Array:
    top = jnp.max(logits)
    weight = jnp.where(jnp.isfinite(logits), jnp.exp(logits - top), 0.0)
    ladder = jnp.cumsum(weight)
    toss = jax.random.uniform(key, shape, ladder.dtype) * ladder[-1]
    index = jnp.searchsorted(ladder, toss, side="right")
    return jnp.clip(index, 0, jnp.shape(logits)[0] - 1).astype(jnp.int32)


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
    selection: PyTree = struct.field(metadata={"axis": None})
    assigned: Array
    due: Array
    slot: Array
    position: Array
    banked: Array
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


def stagger_time_limit(state: PyTree, key: Key, spread: int) -> PyTree:
    if isinstance(state, TimeLimitState):
        elapsed = jax.random.randint(key, jnp.shape(state.time), 0, spread, state.time.dtype)
        state = state.replace(time=elapsed)
    if isinstance(state, WrapperState):
        state = state.replace(env_state=stagger_time_limit(state.env_state, key, spread))
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
        selection: Selector,
        capacity: int = 16384,
        cell_size: int = 4,
        num_probes: int = 8,
        eviction_fn: Callable[[ArchiveState], Array] | None = None,
        eligibility_fn: Callable | None = None,
        identity_fn: Callable | None = None,
        warmup: int = 0,
        gamma: float = 0.99,
        stagger: int = 0,
        reseed_fn: Callable[[PyTree, Key], PyTree] = lambda env_state, key: env_state,
    ):
        super().__init__(env, num_envs=num_envs)
        self.selection = selection
        self.actions = int(getattr(env.action_space(), "num_actions", 1))
        self.archive = Archive(
            archive_size=capacity,
            cell_fn=widen(cell_fn(env=env)),
            eviction_fn=noeviction if eviction_fn is None else eviction_fn,
            cell_size=cell_size,
            num_probes=num_probes,
            identity_fn=None if identity_fn is None else identity_fn(env=env),
        )
        self.eligibility_fn = None if eligibility_fn is None else eligibility_fn(env=env)
        self.warmup = warmup
        self.gamma = gamma
        self.stagger = stagger
        self.reseed_fn = reseed_fn

    def carry(self, shape: PyTree) -> None:
        self.archive = replace(self.archive, carry_shape=shape)

    def ready(self, state) -> Array:
        return jnp.sum(self.archive.eligible(state.archive_state), dtype=jnp.int32) >= self.warmup

    def block(self, state, placed: int) -> Array:
        return (jnp.arange(self.num_envs) >= self.num_envs - placed) & self.ready(state)

    def occupied(self, state, placed: int) -> Array:
        size = self.archive.archive_size
        held = ~self.block(state, placed) & (state.slot >= 0)
        return jnp.zeros(size).at[jnp.where(held, state.slot, size)].add(1.0, mode="drop")

    def eligible(self, env_state: PyTree) -> Array:
        if self.eligibility_fn is None:
            return jnp.ones(self.num_envs, bool)
        return jnp.reshape(self.eligibility_fn(env_state), (self.num_envs, -1)).all(axis=-1)

    def init(self, key: Key):
        env_state, timestep = super().init(key)
        archive_state = self.archive.init(env_state)
        archive_state, index, position, _, evicted = self.archive.add(
            archive_state,
            reset_time_limit(env_state),
            jnp.ones(self.num_envs, bool),
        )
        if self.stagger:
            env_state = stagger_time_limit(env_state, jax.random.fold_in(key, 7), self.stagger)
        state = ArchiveAutoResetState(
            env_state=env_state,
            archive_state=archive_state,
            selection=self.selection.init(self.archive.archive_size, self.actions),
            assigned=jnp.full(self.num_envs, -1, jnp.int32),
            due=jnp.zeros(self.num_envs, bool),
            slot=index,
            position=position,
            banked=position,
            rho=jnp.ones(self.num_envs, bool),
            age=jnp.zeros(self.num_envs, jnp.int32),
            earned=jnp.zeros(self.num_envs),
            graded=jnp.zeros(self.num_envs),
        )
        return state, timestep.replace(
            info=self.mark(
                state,
                archive_state,
                timestep,
                index,
                position,
                jnp.ones(self.num_envs, bool),
                jnp.zeros(self.num_envs, bool),
                evicted,
            ),
        )

    def step(self, key: Key, state, action):
        step_key, fresh_key, place_key, seed_key = jax.random.split(key, 4)
        rho = state.rho
        age = state.age + 1
        env_state, timestep = super().step(step_key, state.env_state, action)
        state = state.replace(env_state=env_state, age=age)
        ended = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        cut = state.due & ~ended
        timestep = timestep.replace(
            truncated=timestep.truncated | broadcast(cut, timestep.truncated)
        )
        done = ended | cut
        warm = done & (state.assigned >= 0)
        fresh, _ = jax.vmap(self._env.init)(jax.random.split(fresh_key, self.num_envs))

        def load_snapshots():
            placed = self.archive.locate(state.archive_state, state.assigned, place_key)
            taken = self.archive.take(state.archive_state, state.assigned, placed)
            restart = jax.tree.map(
                lambda archived, born: jnp.where(broadcast(warm, born), archived, born),
                self.reseed_fn(taken.env_state, seed_key),
                fresh,
            )
            return restart, placed

        restart, placed = jax.lax.cond(
            jnp.any(warm),
            load_snapshots,
            lambda: (fresh, jnp.full(self.num_envs, -1, jnp.int32)),
        )
        state = state.replace(
            env_state=jax.tree.map(
                lambda spawned, live: jnp.where(broadcast(done, live), spawned, live),
                restart,
                state.env_state,
            ),
            rho=jnp.where(done, ~warm, state.rho),
            age=jnp.where(done, 0, state.age),
            due=jnp.zeros_like(state.due),
        )
        timestep = timestep.replace(
            obs=jax.tree.map(
                lambda spawned, live: jnp.where(broadcast(done, live), spawned, live),
                self.observe(state),
                timestep.obs,
            )
        )
        archive_state, index, position, opened, evicted = self.archive.add(
            state.archive_state,
            reset_time_limit(state.env_state),
            ~warm,
            state.rho,
            done | self.eligible(state.env_state),
        )
        if self.archive.carry_shape is not None:
            blank = jax.tree.map(jnp.zeros_like, self.archive.take(archive_state, index, position).carry)
            archive_state = self.archive.stow(
                archive_state, jnp.where(done & ~warm, index, -1), position, blank
            )
        slot = jnp.where(warm, state.assigned, index)
        position = jnp.where(warm, placed, position)
        decay = self.gamma ** (age - 1).astype(jnp.float32)
        reward = timestep.reward.reshape(self.num_envs, -1).sum(axis=-1)
        earned = state.earned + reward
        graded = state.graded + decay * reward

        lox.log(
            {
                "archive/miss": jnp.mean((slot < 0).astype(jnp.float32)),
                "archive/discovery_rate": jnp.sum(opened, dtype=jnp.float32),
                "archive/rho": jnp.sum(archive_state.cell_states.rho, dtype=jnp.float32)
                / jnp.maximum(jnp.sum(archive_state.mask, dtype=jnp.float32), 1.0),
                "archive/num_evictions": jnp.sum(evicted >= 0, dtype=jnp.float32),
            }
        )
        self.recount(earned, graded, age, rho, ended)

        return (
            state.replace(
                archive_state=archive_state,
                slot=slot,
                position=position,
                banked=jnp.where(done, -1, position),
                earned=jnp.where(done, 0.0, earned),
                graded=jnp.where(done, 0.0, graded),
            ),
            timestep.replace(
                info=self.mark(
                    state, archive_state, timestep, slot, position, done & ~warm, done, evicted
                )
            ),
        )

    def mark(self, state, archive_state, timestep, slot, position, start, done, evicted) -> dict:
        info = timestep.info if isinstance(timestep.info, dict) else {}
        info = {**info, "cell": slot, "rho": state.rho, "start": start, "evicted": evicted}
        if self.archive.carry_shape is None:
            return info
        return {
            **info,
            "carry": self.archive.take(archive_state, slot, position).carry,
            "slot": jnp.where(done, slot, -1),
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

    def place(self, state, key: Key, transitions: PyTree):
        update_key, choose_key = jax.random.split(key)
        if transitions is not None:
            state = state.replace(
                selection=self.selection.update(state.selection, update_key, transitions)
            )
        selection, index = self.selection.select(state.selection, self, state, choose_key)
        placed, *_ = jnp.shape(index)
        block = self.block(state, placed)
        assigned = jnp.full(self.num_envs, -1, jnp.int32).at[self.num_envs - placed :].set(index)
        lox.log(
            {
                "archive/placed": jnp.sum(block, dtype=jnp.float32),
                "archive/num_cells": jnp.sum(state.archive_state.mask, dtype=jnp.float32),
                "archive/num_eligible": jnp.sum(
                    self.archive.eligible(state.archive_state), dtype=jnp.float32
                ),
            }
        )
        return state.replace(selection=selection, assigned=jnp.where(block, assigned, -1), due=block)

    def stash(self, state, carry: PyTree):
        return state.replace(
            archive_state=self.archive.stow(state.archive_state, state.slot, state.banked, carry)
        )

    def update(self, state, key: Key, **kwargs):
        return state.replace(env_state=super().update(state.env_state, key, **kwargs))

    def action_mask(self, state) -> Array | None:
        return super().action_mask(state.env_state)

    def observe(self, state) -> PyTree:
        return super().observe(state.env_state)
