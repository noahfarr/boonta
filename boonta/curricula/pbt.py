import jax
import jax.numpy as jnp
import lox
import optax

from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.pbt import MULTIPLIER, PBT, PBTState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, sharded
from boonta.utils.typing import Environment


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


def read_hyperparameters(container: PBT, state: PBTState) -> dict[str, Array]:
    return {
        name: jnp.stack(
            [
                read_value(copy_state, name)
                for copy_state in state.algorithm_state.algorithm_states
            ]
        ).reshape(container.members, container.seeds)
        for name in container.search
    }


def write_hyperparameters(state: PBTState, hyperparameters: dict[str, Array]) -> PBTState:
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


def tally_step(container: PBT, state: PBTState, timestep) -> PBTState:
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


def sample_hyperparameters(container: PBT, state: PBTState) -> PBTState:
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
    container: PBT, state: PBTState, fraction: float, threshold: float, factors: tuple
) -> PBTState:
    key, choice_key, factor_key = jax.random.split(state.key, 3)
    scores, ready = container.fitness(state)
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
    container: PBT, state: PBTState, copied: Array, key: Key, factors: tuple
) -> PBTState:
    perturbed = {}
    for index, (name, values) in enumerate(read_hyperparameters(container, state).items()):
        drawn = jax.random.choice(
            jax.random.fold_in(key, index),
            jnp.array(factors, jnp.float32),
            (container.members, 1),
        )
        perturbed[name] = jnp.where(
            copied.reshape(container.members, 1), values * drawn, values
        )
    return write_hyperparameters(state, perturbed)


def report_members(container: PBT, state: PBTState) -> None:
    scores, ready = container.fitness(state)
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
) -> tuple[PBT, Environment, Pit, Lap]:
    container = PBT(
        algorithm, members=members, seeds=seeds, window=window, search=search
    )
    factors = tuple(float(factor) for factor in factors)

    def manage(node: PBTState) -> PBTState:
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
            if isinstance(node, PBTState):
                return change(node)
            return node

        return apply

    def pit(state):
        algorithm_state = jax.tree.map(
            visit(manage),
            state.algorithm_state,
            is_leaf=lambda node: isinstance(node, PBTState),
        )
        return state.replace(algorithm_state=algorithm_state)

    def lap(state):
        algorithm_state = jax.tree.map(
            visit(lambda node: tally_step(container, node, state.timestep)),
            state.algorithm_state,
            is_leaf=lambda node: isinstance(node, PBTState),
        )
        return state.replace(algorithm_state=algorithm_state)

    return container, environment, pit, lap
