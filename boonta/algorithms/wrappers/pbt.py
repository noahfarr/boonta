from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (Array, Key, PyTree, Timestep, Transition,
                          canonicalize_dtype, sharded)

from ..algorithm import Algorithm
from .population import Population, PopulationState
from .wrapper import Wrapper, WrapperState


def attach_multiplier(algorithm: Algorithm) -> Algorithm:
    names = {field.name for field in fields(algorithm)}
    if "optimizer" not in names:
        optimizers = sorted(name for name in names if name.endswith("optimizer"))
        raise ValueError(
            f"PBT scales the learning rate through a multiplier chained onto the "
            f"algorithm's `optimizer`, but {type(algorithm).__name__} has "
            f"{optimizers or 'no optimizer field'} instead of a single `optimizer`. "
            f"PBT supports single-optimizer algorithms only."
        )
    return replace(
        algorithm,
        optimizer=optax.chain(
            algorithm.optimizer, optax.inject_hyperparams(optax.scale)(step_size=1.0)
        ),
    )


def read_multiplier(algorithm_state) -> Array:
    _, injected = algorithm_state.optimizer_state
    return injected.hyperparams["step_size"]


def write_multiplier(algorithm_state, multiplier: Array):
    rest, injected = algorithm_state.optimizer_state
    current = injected.hyperparams["step_size"]
    hyperparams = {
        **injected.hyperparams,
        "step_size": jnp.asarray(multiplier, current.dtype),
    }
    return algorithm_state.replace(
        optimizer_state=(rest, injected._replace(hyperparams=hyperparams))
    )


def finish_episodes(running: Array, reward: Array, done: Array):
    def accumulate(running, entry):
        earned, finished = entry
        running = running + earned
        return jnp.where(finished, 0.0, running), running

    running, totals = jax.lax.scan(
        accumulate, running, (reward.astype(jnp.float32), done)
    )
    return running, totals


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


@struct.dataclass(frozen=True)
class PBTState(WrapperState):
    algorithm_state: PopulationState
    running: Array = struct.field(metadata={"axis": "data"})
    window: Array
    filled: Array
    replaced: Array
    iteration: Array
    step: Array


@dataclass
class PBT(Wrapper):
    members: int
    seeds: int
    interval: int
    fraction: float
    threshold: float
    window: int
    low: float
    high: float
    factors: tuple[float, ...]

    def __post_init__(self):
        self.algorithm = Population(
            attach_multiplier(self.algorithm), count=self.members * self.seeds
        )

    @property
    def copies(self) -> int:
        return self.members * self.seeds

    def multipliers(self, state: PBTState) -> Array:
        return jnp.stack(
            [
                read_multiplier(algorithm_state)
                for algorithm_state in state.algorithm_state.algorithm_states
            ]
        ).reshape(self.members, self.seeds)

    def assign(self, state: PBTState, multipliers: Array) -> PBTState:
        flat = multipliers.reshape(-1)
        algorithm_states = tuple(
            write_multiplier(algorithm_state, jnp.take(flat, index))
            for index, algorithm_state in enumerate(
                state.algorithm_state.algorithm_states
            )
        )
        return state.replace(
            algorithm_state=state.algorithm_state.replace(
                algorithm_states=algorithm_states
            )
        )

    def fitness(self, state: PBTState) -> tuple[Array, Array]:
        scores = state.window.mean(axis=1).reshape(self.members, self.seeds)
        ready = (state.filled >= self.window).reshape(self.members, self.seeds)
        return scores, ready.all(axis=1)

    def init(self, key: Key, timestep: Timestep) -> PBTState:
        population_key, multiplier_key = jax.random.split(key)
        population_state = self.algorithm.init(population_key, timestep)
        state = PBTState(
            algorithm_state=population_state,
            running=jnp.zeros(timestep.reward.shape, jnp.float32),
            window=jnp.zeros((self.copies, self.window), jnp.float32),
            filled=jnp.zeros(self.copies, jnp.int32),
            replaced=jnp.zeros(self.members, jnp.int32),
            iteration=jnp.array(0, jnp.int32),
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )
        exponents = jax.random.uniform(
            multiplier_key,
            (self.members, 1),
            minval=jnp.log(self.low),
            maxval=jnp.log(self.high),
        )
        return self.assign(
            state, jnp.broadcast_to(jnp.exp(exponents), (self.members, self.seeds))
        )

    def synchronize(self, state: PBTState) -> PBTState:
        return state.replace(
            algorithm_state=self.algorithm.synchronize(
                state.algorithm_state.replace(step=state.step)
            )
        )

    def step(self, state: PBTState, key: Key, timestep: Timestep, temperature=1.0):
        state = self.synchronize(state)
        population_state, action, aux = self.algorithm.step(
            state.algorithm_state, key, timestep, temperature
        )
        return state.replace(algorithm_state=population_state), action, aux

    def tally(self, state: PBTState, transitions: Transition) -> PBTState:
        done = transitions.second.terminated | transitions.second.truncated
        running, totals = finish_episodes(
            state.running, transitions.second.reward, done
        )
        steps, *_ = done.shape

        def split(leaf):
            return jnp.moveaxis(
                leaf.reshape(steps, self.copies, -1), 1, 0
            ).reshape(self.copies, -1)

        window, filled = jax.vmap(record_episodes)(
            state.window, state.filled, split(totals), split(done)
        )
        return state.replace(running=running, window=window, filled=filled)

    def exploit(self, state: PBTState, key: Key) -> PBTState:
        choice_key, factor_key = jax.random.split(key)
        scores, ready = self.fitness(state)
        sources, copied = choose_sources(
            scores, ready, choice_key, self.fraction, self.threshold
        )
        copies = (
            sources.reshape(self.members, 1) * self.seeds + jnp.arange(self.seeds)
        ).reshape(-1)
        state = state.replace(
            algorithm_state=state.algorithm_state.replace(
                algorithm_states=gather_copies(
                    state.algorithm_state.algorithm_states, copies
                )
            )
        )
        factors = jax.random.choice(
            factor_key, jnp.array(self.factors, jnp.float32), (self.members, 1)
        )
        multipliers = self.multipliers(state)
        state = self.assign(
            state,
            jnp.where(
                copied.reshape(self.members, 1), multipliers * factors, multipliers
            ),
        )
        reset = jnp.repeat(copied, self.seeds)
        return state.replace(
            filled=jnp.where(reset, 0, state.filled),
            replaced=state.replaced + copied.astype(state.replaced.dtype),
        )

    def report(self, state: PBTState) -> None:
        scores, ready = self.fitness(state)
        multipliers = self.multipliers(state)
        fitness = jnp.where(ready, scores.mean(axis=1), jnp.nan)
        logs = {}
        for member in range(self.members):
            logs |= {
                f"pbt/member_{member}/multiplier": jnp.take(multipliers, member, axis=0).mean(),
                f"pbt/member_{member}/fitness": jnp.take(fitness, member),
                f"pbt/member_{member}/replaced": jnp.take(state.replaced, member),
            }
        lox.log(logs)

    def update(self, state: PBTState, key: Key, transitions: Transition) -> PBTState:
        update_key, exploit_key = jax.random.split(key)
        state = self.synchronize(state)
        state = state.replace(
            algorithm_state=self.algorithm.update(
                state.algorithm_state, update_key, transitions
            )
        )
        state = self.tally(state, transitions)
        due = jnp.mod(state.iteration + 1, self.interval) == 0
        state = jax.lax.cond(
            due,
            lambda state: self.exploit(state, exploit_key),
            lambda state: state,
            state,
        )
        self.report(state)
        return state.replace(iteration=state.iteration + 1)
