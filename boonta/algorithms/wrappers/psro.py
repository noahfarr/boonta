from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Protocol

import jax
import jax.numpy as jnp
import lox
from flax import struct
from boonta.utils import (Array, Key, PyTree, Timestep, Transition,
                          canonicalize_dtype, conditional_update, place)

from ..algorithm import Algorithm
from ..auxiliary_losses import Anchor
from .ensemble import Ensemble, EnsembleState
from .wrapper import Wrapper, WrapperState


@dataclass
class Meta:
    payoff: Array
    counts: Array
    lineages: Array
    members: Array
    learners: int
    index: int
    iteration: Array
    names: tuple[str, ...]

    def descends(self, lineage: str) -> Array:
        return self.lineages == self.names.index(lineage)


def winrate(meta: Meta) -> Array:
    row = jnp.take(meta.payoff, meta.index, axis=0)
    counts = jnp.take(meta.counts, meta.index, axis=0)
    return (row / jnp.maximum(counts, 1.0) + 1.0) / 2.0


class Solver(Protocol):
    def __call__(self, meta: Meta) -> Array: ...


def self_play() -> Solver:
    def solver(meta: Meta) -> Array:
        return jax.nn.one_hot(meta.index, meta.members.shape[0])

    return solver


def latest() -> Solver:
    def solver(meta: Meta) -> Array:
        slots = jnp.arange(meta.members.shape[0])
        newest = jnp.max(jnp.where(meta.members > 0, slots, -1))
        return jax.nn.one_hot(newest, meta.members.shape[0])

    return solver


def fsp() -> Solver:
    def solver(meta: Meta) -> Array:
        weight = meta.members.astype(jnp.float32)
        return weight / weight.sum()

    return solver


def weighted(meta: Meta, eligible: Array, weighting: Callable) -> Array:
    weight = jnp.where(eligible, weighting(winrate(meta)), 0.0)
    total = weight.sum()
    uniform = meta.members / meta.members.sum()
    return jnp.where(total > 0, weight / jnp.maximum(total, 1e-8), uniform)


def pfsp(weighting: Callable[[Array], Array]) -> Solver:
    def solver(meta: Meta) -> Array:
        return weighted(meta, meta.members > 0, weighting)

    return solver


def pfsp_vs_lineage(
    lineage: str, weighting: Callable[[Array], Array], current: bool = False
) -> Solver:
    def solver(meta: Meta) -> Array:
        eligible = (meta.members > 0) & meta.descends(lineage)
        if current:
            slots = jnp.arange(meta.members.shape[0])
            eligible = eligible & (slots < meta.learners)
        return weighted(meta, eligible, weighting)

    return solver


def forgotten(
    lineage: str, weighting: Callable[[Array], Array], level: float = 0.7
) -> Solver:
    def solver(meta: Meta) -> Array:
        eligible = (meta.members > 0) & meta.descends(lineage) & (winrate(meta) < level)
        return weighted(meta, eligible, weighting)

    return solver


def nash(iterations: int = 128) -> Solver:
    def solver(meta: Meta) -> Array:
        payoff = meta.payoff / jnp.maximum(meta.counts, 1.0)
        valid = meta.members > 0

        def play(counts, _):
            rows, cols = counts
            gain = payoff @ (cols / cols.sum())
            loss = (rows / rows.sum()) @ payoff
            row = jnp.argmax(jnp.where(valid, gain, -jnp.inf))
            col = jnp.argmin(jnp.where(valid, loss, jnp.inf))
            return (rows.at[row].add(1.0), cols.at[col].add(1.0)), None

        counts = (valid.astype(jnp.float32), valid.astype(jnp.float32))
        (_, cols), _ = jax.lax.scan(play, counts, None, length=iterations)
        return cols / cols.sum()

    return solver


def ladder(primary: Solver, fallback: Solver, level: float = 0.2) -> Solver:
    def solver(meta: Meta) -> Array:
        distribution = primary(meta)
        losing = jnp.sum(distribution * winrate(meta)) < level
        return jnp.where(losing, fallback(meta), distribution)

    return solver


def mixture(solvers: list[Solver], weights: list[float]) -> Solver:
    def solver(meta: Meta) -> Array:
        distribution = sum(
            weight * component(meta) for component, weight in zip(solvers, weights)
        )
        return distribution / distribution.sum()

    return solver


def hard(exponent: float = 1.0) -> Callable[[Array], Array]:
    return lambda winrate: jnp.maximum(1.0 - winrate, 1e-8) ** exponent


def even() -> Callable[[Array], Array]:
    return lambda winrate: winrate * (1.0 - winrate)


class Admit(Protocol):
    def __call__(self, meta: Meta) -> Array: ...


def always() -> Admit:
    def admit(meta: Meta) -> Array:
        return jnp.array(True)

    return admit


def threshold(level: float = 0.7, minimum: float = 1.0) -> Admit:
    def admit(meta: Meta) -> Array:
        played = jnp.take(meta.counts, meta.index, axis=0) > minimum
        rate = jnp.sum(jnp.where(played, winrate(meta), 0.0)) / jnp.maximum(
            played.sum(), 1.0
        )
        return (played.sum() > 0) & (rate > level)

    return admit


def periodic(every: int) -> Admit:
    def admit(meta: Meta) -> Array:
        return jnp.mod(meta.iteration + 1, every) == 0

    return admit


def either(*admits: Admit) -> Admit:
    def admit(meta: Meta) -> Array:
        return reduce(
            lambda decision, candidate: decision | candidate(meta),
            admits,
            jnp.array(False),
        )

    return admit


@dataclass
class Learner:
    lineage: str
    solver: Solver
    admit: Admit
    resets: bool = False
    restart: Solver | None = None
    kl_coefficient: float = 0.0
    warmstart: str | None = None


@dataclass
class Member:
    lineage: str
    params: PyTree


@struct.dataclass(frozen=True)
class LearnerState:
    initial_state: PyTree
    opponent: Array
    returns: Array = struct.field(metadata={"axis": "data"})


@struct.dataclass(frozen=True)
class PSROState(WrapperState):
    algorithm_state: EnsembleState
    learners: tuple
    population: PyTree
    payoff: Array
    counts: Array
    members: Array
    lineages: Array
    cursor: Array
    iteration: Array
    step: Array

    @property
    def params(self) -> PyTree:
        return self.algorithm_state.params


@dataclass
class PSRO(Wrapper):
    learners: list[Learner]
    capacity: int
    decay: float
    population: tuple[Member, ...] = ()
    warmstarts: dict[int, PyTree] = field(default_factory=dict)
    dtype: jnp.dtype = jnp.bfloat16

    @property
    def oracle(self) -> Algorithm:
        return self.algorithm.algorithm

    @property
    def names(self) -> tuple[str, ...]:
        lineages = [learner.lineage for learner in self.learners] + [
            member.lineage for member in self.population
        ]
        return tuple(dict.fromkeys(lineages))

    def __post_init__(self):
        self.algorithm = Ensemble(self.algorithm, count=len(self.learners))
        occupied = len(self.learners) + len(self.population)
        assert occupied <= self.capacity, (
            f"capacity ({self.capacity}) must hold the {len(self.learners)} learners "
            f"and the {len(self.population)} members of the initial population"
        )
        if any(learner.kl_coefficient > 0.0 for learner in self.learners):
            assert hasattr(self.oracle, "auxiliary_losses"), (
                f"{type(self.oracle).__name__} takes no auxiliary losses, but some "
                f"learners set kl_coefficient > 0 to stay near their initial "
                f"parameters. Use kl_coefficient=0.0 for every learner, or an oracle "
                f"that can be anchored."
            )

    def meta(self, state: PSROState, index: int) -> Meta:
        return Meta(
            payoff=state.payoff,
            counts=state.counts,
            lineages=state.lineages,
            members=state.members,
            learners=len(self.learners),
            index=index,
            iteration=state.iteration,
            names=self.names,
        )

    def read(self, population: PyTree, slot: Array, like: PyTree | None = None):
        if like is None:
            return jax.tree.map(lambda leaf: jnp.take(leaf, slot, axis=0), population)
        return jax.tree.map(
            lambda leaf, reference: jnp.take(leaf, slot, axis=0).astype(
                reference.dtype
            ),
            population,
            like,
        )

    def write(self, population: PyTree, slot, params: PyTree, when=None):
        def put(leaf, value):
            value = value.astype(leaf.dtype)
            if when is None:
                return leaf.at[slot].set(value)
            return leaf.at[slot].set(
                jnp.where(when, value, jnp.take(leaf, slot, axis=0))
            )

        return jax.tree.map(put, population, params)

    def opponents(self, state: PSROState) -> PyTree:
        drawn = [
            self.read(state.population, learner.opponent) for learner in state.learners
        ]
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *drawn)

    def tally(self, matrix: Array, index: int, rival: Array, value, sign: float):
        return (
            matrix.at[index]
            .multiply(self.decay)
            .at[:, index]
            .multiply(self.decay)
            .at[index, rival]
            .add(value)
            .at[rival, index]
            .add(sign * value)
        )

    def score(self, learner_state: LearnerState, view: Transition):
        done = view.second.terminated | view.second.truncated

        def accumulate(carry, entry):
            reward, finished = entry
            carry = carry + reward
            return jnp.where(finished, 0.0, carry), (
                jnp.where(finished, carry, 0.0),
                finished.astype(carry.dtype),
            )

        returns, (scores, finishes) = jax.lax.scan(
            accumulate, learner_state.returns, (view.second.reward, done)
        )
        return returns, scores.sum(), finishes.sum()

    def anchored(self, learner_state: LearnerState, learner: Learner) -> Algorithm:
        if learner.kl_coefficient <= 0.0:
            return self.oracle
        return replace(
            self.oracle,
            auxiliary_losses=(
                *self.oracle.auxiliary_losses,
                Anchor(learner_state.initial_state.params, learner.kl_coefficient),
            ),
        )

    def report(self, index: int, meta: Meta, score, count, admitted) -> None:
        lox.log(
            {
                f"learner_{index}/episode_return": score / jnp.maximum(count, 1.0),
                f"learner_{index}/episodes": count,
                f"learner_{index}/admitted": admitted.astype(jnp.float32),
            }
        )
        if self.population:
            reference = winrate(meta)[
                len(self.learners) : len(self.learners) + len(self.population)
            ]
            lox.log({f"learner_{index}/reference": reference.mean()})

    def init(self, key: Key, timestep: Timestep) -> PSROState:
        parts = len(self.learners)
        ensemble_state = self.algorithm.init(key, timestep)
        ensemble_state = ensemble_state.replace(
            algorithm_states=tuple(
                state.replace(params=self.warmstarts.get(index, state.params))
                for index, state in enumerate(ensemble_state.algorithm_states)
            )
        )
        width = timestep.terminated.shape[0] // parts
        states = tuple(
            LearnerState(
                initial_state=algorithm_state,
                opponent=jnp.array(index),
                returns=jnp.zeros((width,)),
            )
            for index, algorithm_state in enumerate(ensemble_state.algorithm_states)
        )
        stacked = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *(state.params for state in ensemble_state.algorithm_states),
        )
        population = jax.tree.map(
            lambda leaf: jnp.zeros((self.capacity, *leaf.shape[1:]), self.dtype)
            .at[:parts]
            .set(leaf.astype(self.dtype)),
            stacked,
        )
        population = reduce(
            lambda pool, entry: self.write(pool, parts + entry[0], entry[1].params),
            enumerate(self.population),
            population,
        )
        occupied = parts + len(self.population)
        lineages = jnp.array(
            [self.names.index(learner.lineage) for learner in self.learners]
            + [self.names.index(member.lineage) for member in self.population]
        )
        return PSROState(
            algorithm_state=ensemble_state,
            learners=states,
            population=population,
            payoff=jnp.zeros((self.capacity, self.capacity)),
            counts=jnp.zeros((self.capacity, self.capacity)),
            members=jnp.zeros(self.capacity).at[:occupied].set(1.0),
            lineages=jnp.full(self.capacity, -1).at[:occupied].set(lineages),
            cursor=jnp.array(occupied),
            iteration=jnp.array(0),
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )

    def synchronize(self, state: PSROState) -> PSROState:
        return state.replace(
            algorithm_state=self.algorithm.synchronize(
                state.algorithm_state.replace(step=state.step)
            )
        )

    def step(self, state: PSROState, key: Key, timestep: Timestep, temperature=1.0):
        state = self.synchronize(state)
        ensemble_state, action, aux = self.algorithm.step(
            state.algorithm_state, key, timestep, temperature
        )
        return state.replace(algorithm_state=ensemble_state), action, aux

    def update(self, state: PSROState, key: Key, transitions: Transition) -> PSROState:
        state = self.synchronize(state)
        for index, learner in enumerate(self.learners):
            state = self.iterate(
                state, index, learner, transitions, jax.random.fold_in(key, index)
            )
        population = reduce(
            lambda pool, entry: self.write(pool, entry[0], entry[1].params),
            enumerate(state.algorithm_state.algorithm_states),
            state.population,
        )
        return state.replace(population=population, iteration=state.iteration + 1)

    def iterate(
        self, state: PSROState, index: int, learner: Learner, transitions, key: Key
    ) -> PSROState:
        response_key, sample_key, restart_key = jax.random.split(key, 3)
        learner_state = state.learners[index]
        view = self.algorithm.portion(transitions, index)

        returns, score, count = self.score(learner_state, view)
        rival = learner_state.opponent
        state = state.replace(
            payoff=self.tally(state.payoff, index, rival, score, -1.0),
            counts=self.tally(state.counts, index, rival, count, 1.0),
        )

        responded = self.algorithm.respond(
            state.algorithm_state,
            index,
            response_key,
            view,
            self.anchored(learner_state, learner),
        )
        inner = responded.algorithm_states[index]
        admitted = learner.admit(self.meta(state, index)) & (
            state.cursor < self.capacity
        )
        population = self.write(
            state.population, state.cursor, inner.params, when=admitted
        )
        if learner.resets:
            reset_state = learner_state.initial_state
            if learner.restart:
                slot = jax.random.categorical(
                    restart_key, jnp.log(learner.restart(self.meta(state, index)))
                )
                reset_state = reset_state.replace(
                    params=self.read(population, slot, like=reset_state.params)
                )
            inner = conditional_update(reset_state, inner, admitted)

        self.report(index, self.meta(state, index), score, count, admitted)
        state = state.replace(
            algorithm_state=responded.replace(
                algorithm_states=place(responded.algorithm_states, index, inner)
            ),
            population=population,
            members=state.members.at[state.cursor].set(
                jnp.where(admitted, 1.0, jnp.take(state.members, state.cursor))
            ),
            lineages=state.lineages.at[state.cursor].set(
                jnp.where(
                    admitted,
                    self.names.index(learner.lineage),
                    jnp.take(state.lineages, state.cursor),
                )
            ),
            cursor=state.cursor + admitted.astype(state.cursor.dtype),
        )
        return state.replace(
            learners=place(
                state.learners,
                index,
                learner_state.replace(
                    returns=returns,
                    opponent=jax.random.categorical(
                        sample_key, jnp.log(learner.solver(self.meta(state, index)))
                    ),
                ),
            )
        )
