from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.injected import Injected
from boonta.algorithms.wrappers.population import Population, PopulationState
from boonta.algorithms.wrappers.wrapper import Wrapper, WrapperState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import (Array, Key, Timestep, Transition, canonicalize_dtype,
                          sharded)
from boonta.utils.typing import Environment

MULTIPLIER = "learning_rate_multiplier"


def scale_by_multiplier(multiplier: float) -> optax.GradientTransformation:
    return optax.scale(multiplier)


def attach_multiplier(algorithm: Algorithm) -> Algorithm:
    optimizers = [
        field.name
        for field in fields(algorithm)
        if isinstance(getattr(algorithm, field.name), optax.GradientTransformation)
    ]
    if not optimizers:
        raise ValueError(
            f"PBT scales the learning rate through a multiplier chained onto every "
            f"optax.GradientTransformation the algorithm holds, but "
            f"{type(algorithm).__name__} holds none."
        )
    return replace(
        algorithm,
        **{
            name: optax.chain(
                getattr(algorithm, name),
                optax.inject_hyperparams(scale_by_multiplier)(multiplier=1.0),
            )
            for name in optimizers
        },
    )


@struct.dataclass(frozen=True)
class TrialsState(WrapperState):
    algorithm_state: PopulationState
    running: Array = struct.field(metadata={"axis": "data"})
    window: Array
    filled: Array
    replaced: Array
    updates: Array
    handled: Array
    seeded: Array
    key: Key
    step: Array


@dataclass
class Trials(Wrapper):
    members: int
    seeds: int
    window: int
    search: dict

    def __post_init__(self):
        self.search = {
            name: tuple(float(bound) for bound in bounds)
            for name, bounds in dict(self.search).items()
        }
        algorithm = self.algorithm
        if MULTIPLIER in self.search:
            algorithm = attach_multiplier(algorithm)
        injected = Injected(
            algorithm, names=tuple(name for name in self.search if name != MULTIPLIER)
        )
        self.algorithm = Population(injected, count=self.copies)

    @property
    def copies(self) -> int:
        return self.members * self.seeds

    def init(self, key: Key, timestep: Timestep) -> TrialsState:
        population_key, search_key = jax.random.split(key)
        return TrialsState(
            algorithm_state=self.algorithm.init(population_key, timestep),
            running=jnp.zeros(timestep.reward.shape, jnp.float32),
            window=jnp.zeros((self.copies, self.window), jnp.float32),
            filled=jnp.zeros(self.copies, jnp.int32),
            replaced=jnp.zeros(self.members, jnp.int32),
            updates=jnp.array(0, jnp.int32),
            handled=jnp.array(0, jnp.int32),
            seeded=jnp.array(False),
            key=search_key,
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )

    def synchronize(self, state: TrialsState) -> TrialsState:
        return state.replace(
            algorithm_state=self.algorithm.synchronize(
                state.algorithm_state.replace(step=state.step)
            )
        )

    def step(self, state: TrialsState, key: Key, timestep: Timestep, temperature=1.0):
        state = self.synchronize(state)
        population_state, action, aux = self.algorithm.step(
            state.algorithm_state, key, timestep, temperature
        )
        return state.replace(algorithm_state=population_state), action, aux

    def update(
        self, state: TrialsState, key: Key, transitions: Transition
    ) -> TrialsState:
        state = self.synchronize(state)
        return state.replace(
            algorithm_state=self.algorithm.update(
                state.algorithm_state, key, transitions
            ),
            updates=state.updates + 1,
        )


def member_fitness(container: Trials, state: TrialsState) -> tuple[Array, Array]:
    scores = state.window.mean(axis=1).reshape(container.members, container.seeds)
    ready = (state.filled >= container.window).reshape(
        container.members, container.seeds
    )
    return scores, ready.all(axis=1)


def leading_member(container: Trials, state: TrialsState) -> Array:
    scores, ready = member_fitness(container, state)
    mean = scores.mean(axis=1)
    return jnp.where(
        ready.any(),
        jnp.argmax(jnp.where(ready, mean, -jnp.inf)),
        jnp.argmax(mean),
    )


def holds_multiplier(node) -> bool:
    return (
        isinstance(node, optax.InjectStatefulHyperparamsState)
        and "multiplier" in node.hyperparams
    )


def read_multiplier(algorithm_state) -> Array:
    first, *_ = [
        node
        for node in jax.tree.leaves(algorithm_state, is_leaf=holds_multiplier)
        if holds_multiplier(node)
    ]
    return first.hyperparams["multiplier"]


def write_multiplier(algorithm_state, multiplier: Array):
    def overwrite(node):
        if not holds_multiplier(node):
            return node
        current = node.hyperparams["multiplier"]
        return node._replace(
            hyperparams={
                **node.hyperparams,
                "multiplier": jnp.asarray(multiplier, current.dtype),
            }
        )

    return jax.tree.map(overwrite, algorithm_state, is_leaf=holds_multiplier)


def read_value(copy_state, name: str) -> Array:
    if name == MULTIPLIER:
        return read_multiplier(copy_state.algorithm_state)
    return copy_state.values[name]


def write_value(copy_state, name: str, value: Array):
    if name == MULTIPLIER:
        return copy_state.replace(
            algorithm_state=write_multiplier(copy_state.algorithm_state, value)
        )
    current = copy_state.values[name]
    return copy_state.replace(
        values={**copy_state.values, name: jnp.asarray(value, current.dtype)}
    )


def read_hyperparameters(container: Trials, state: TrialsState) -> dict[str, Array]:
    return {
        name: jnp.stack(
            [
                read_value(copy_state, name)
                for copy_state in state.algorithm_state.algorithm_states
            ]
        ).reshape(container.members, container.seeds)
        for name in container.search
    }


def write_hyperparameters(state: TrialsState, hyperparameters: dict[str, Array]) -> TrialsState:
    flat = {name: values.reshape(-1) for name, values in hyperparameters.items()}
    copy_states = []
    for index, copy_state in enumerate(state.algorithm_state.algorithm_states):
        for name, values in flat.items():
            copy_state = write_value(copy_state, name, jnp.take(values, index))
        copy_states.append(copy_state)
    return state.replace(
        algorithm_state=state.algorithm_state.replace(
            algorithm_states=tuple(copy_states)
        )
    )


def record_episodes(window: Array, filled: Array, returns: Array, finished: Array):
    (size,) = window.shape
    rank = jnp.cumsum(finished) - 1
    total = finished.sum()
    keep = finished & (rank >= total - size)
    slot = jnp.where(keep, (filled + rank) % size, size)
    return window.at[slot].set(returns, mode="drop"), filled + total


def choose_sources(
    fitness: Array, ready: Array, key: Key, fraction: float, threshold: float
) -> tuple[Array, Array]:
    members, seeds = fitness.shape
    count = max(1, int(fraction * members))
    mean = fitness.mean(axis=1)
    if seeds > 1:
        variance = fitness.var(axis=1, ddof=1)
    else:
        variance = jnp.zeros(members, fitness.dtype)
    ascending = jnp.argsort(jnp.argsort(jnp.where(ready, mean, jnp.inf)))
    descending = jnp.argsort(jnp.argsort(jnp.where(ready, -mean, jnp.inf)))
    loser = ready & (ascending < count)
    winner = ready & (descending < count)
    picks = jax.random.categorical(
        key, jnp.where(winner, 0.0, -jnp.inf), shape=(members,)
    )
    gap = jnp.take(mean, picks) - mean
    error = jnp.sqrt((jnp.take(variance, picks) + variance) / seeds)
    copied = loser & jnp.take(winner, picks) & (gap > threshold * error)
    return jnp.where(copied, picks, jnp.arange(members)), copied


def gather_copies(algorithm_states: tuple, sources: Array) -> tuple:
    first, *_ = algorithm_states
    axes = sharded(first)
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *algorithm_states)

    def adopt(index, algorithm_state):
        source = jnp.take(sources, index)

        def inherit(axis, own, gathered):
            if axis:
                return own
            return jnp.take(gathered, source, axis=0)

        return jax.tree.map(inherit, axes, algorithm_state, stacked)

    return tuple(
        adopt(index, algorithm_state)
        for index, algorithm_state in enumerate(algorithm_states)
    )


def tally_step(container: Trials, state: TrialsState, timestep) -> TrialsState:
    done = timestep.terminated | timestep.truncated
    totals = state.running + timestep.reward.astype(jnp.float32)
    window, filled = jax.vmap(record_episodes)(
        state.window,
        state.filled,
        totals.reshape(container.copies, -1),
        done.reshape(container.copies, -1),
    )
    return state.replace(
        running=jnp.where(done, 0.0, totals), window=window, filled=filled
    )


def sample_hyperparameters(container: Trials, state: TrialsState) -> TrialsState:
    key, search_key = jax.random.split(state.key)
    hyperparameters = {}
    for index, (name, (low, high)) in enumerate(container.search.items()):
        exponents = jax.random.uniform(
            jax.random.fold_in(search_key, index),
            (container.members, 1),
            minval=jnp.log(low),
            maxval=jnp.log(high),
        )
        hyperparameters[name] = jnp.broadcast_to(
            jnp.exp(exponents), (container.members, container.seeds)
        )
    return write_hyperparameters(state, hyperparameters).replace(
        running=jnp.zeros_like(state.running),
        filled=jnp.zeros_like(state.filled),
        seeded=jnp.array(True),
        key=key,
    )


def exploit_members(
    container: Trials, state: TrialsState, fraction: float, threshold: float, factors: tuple
) -> TrialsState:
    key, choice_key, factor_key = jax.random.split(state.key, 3)
    scores, ready = member_fitness(container, state)
    sources, copied = choose_sources(scores, ready, choice_key, fraction, threshold)
    copies = (
        sources.reshape(container.members, 1) * container.seeds
        + jnp.arange(container.seeds)
    ).reshape(-1)
    state = state.replace(
        algorithm_state=state.algorithm_state.replace(
            algorithm_states=gather_copies(state.algorithm_state.algorithm_states, copies)
        )
    )
    state = explore_members(container, state, copied, factor_key, factors)
    return state.replace(
        filled=jnp.where(jnp.repeat(copied, container.seeds), 0, state.filled),
        replaced=state.replaced + copied.astype(state.replaced.dtype),
        key=key,
    )


def explore_members(
    container: Trials, state: TrialsState, copied: Array, key: Key, factors: tuple
) -> TrialsState:
    perturbed = {}
    for index, (name, values) in enumerate(read_hyperparameters(container, state).items()):
        drawn = jax.random.choice(
            jax.random.fold_in(key, index),
            jnp.array(factors, jnp.float32),
            (container.members, 1),
        )
        low, high = container.search[name]
        perturbed[name] = jnp.clip(
            jnp.where(copied.reshape(container.members, 1), values * drawn, values),
            low,
            high,
        )
    return write_hyperparameters(state, perturbed)


def report_members(container: Trials, state: TrialsState) -> None:
    scores, ready = member_fitness(container, state)
    fitness = jnp.where(ready, scores.mean(axis=1), jnp.nan)
    hyperparameters = read_hyperparameters(container, state)
    logs = {}
    for member in range(container.members):
        logs |= {
            f"pbt/member_{member}/fitness": jnp.take(fitness, member),
            f"pbt/member_{member}/replaced": jnp.take(state.replaced, member),
        }
        for name, values in hyperparameters.items():
            logs[f"pbt/member_{member}/{name}"] = jnp.take(values, member, axis=0).mean()
    lox.log(logs, tags=("training",))


def pbt(
    algorithm: Algorithm,
    environment: Environment,
    members: int,
    seeds: int,
    interval: int,
    fraction: float,
    threshold: float,
    window: int,
    search,
    factors,
    **kwargs,
) -> tuple[Trials, Environment, Pit, Lap]:
    container = Trials(
        algorithm, members=members, seeds=seeds, window=window, search=search
    )
    factors = tuple(float(factor) for factor in factors)

    def manage(node: TrialsState) -> TrialsState:
        node = jax.lax.cond(
            node.seeded,
            lambda node: node,
            lambda node: sample_hyperparameters(container, node),
            node,
        )
        due = (node.updates > node.handled) & (jnp.mod(node.updates, interval) == 0)
        node = jax.lax.cond(
            due,
            lambda node: exploit_members(container, node, fraction, threshold, factors),
            lambda node: node,
            node,
        )
        report_members(container, node)
        return node.replace(handled=node.updates)

    def visit(change):
        def apply(node):
            if isinstance(node, TrialsState):
                return change(node)
            return node

        return apply

    def pit(state):
        algorithm_state = jax.tree.map(
            visit(manage),
            state.algorithm_state,
            is_leaf=lambda node: isinstance(node, TrialsState),
        )
        return state.replace(algorithm_state=algorithm_state)

    def lap(state):
        algorithm_state = jax.tree.map(
            visit(lambda node: tally_step(container, node, state.timestep)),
            state.algorithm_state,
            is_leaf=lambda node: isinstance(node, TrialsState),
        )
        return state.replace(algorithm_state=algorithm_state)

    return container, environment, pit, lap
