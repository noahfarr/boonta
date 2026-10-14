from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import reduce
from typing import Protocol

import jax
import jax.numpy as jnp
import lox
from flax import struct
from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.roster import Roster, RosterState
from boonta.environments.wrappers import Opponent, OpponentState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Transition, conditional_update, take
from boonta.utils.typing import Environment


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
    opponent: Array
    returns: Array = struct.field(metadata={"axis": "data"})
    resets: Array


@struct.dataclass(frozen=True)
class PSROState:
    learners: tuple
    pool: PyTree
    payoff: Array
    counts: Array
    members: Array
    lineages: Array
    cursor: Array
    iteration: Array


@dataclass
class PSRO(Pit):
    roster: Roster
    opponent: Opponent
    learners: list[Learner]
    capacity: int
    decay: float
    pool: tuple[Member, ...] = ()
    dress: Callable = lambda state: state
    dtype: jnp.dtype = jnp.bfloat16

    @property
    def names(self) -> tuple[str, ...]:
        lineages = [learner.lineage for learner in self.learners] + [
            member.lineage for member in self.pool
        ]
        return tuple(dict.fromkeys(lineages))

    def __post_init__(self):
        occupied = len(self.learners) + len(self.pool)
        assert occupied <= self.capacity, (
            f"capacity ({self.capacity}) must hold the {len(self.learners)} learners "
            f"and the {len(self.pool)} members of the initial pool"
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

    def read(self, pool: PyTree, slot: Array, like: PyTree | None = None):
        if like is None:
            return jax.tree.map(lambda leaf: jnp.take(leaf, slot, axis=0), pool)
        return jax.tree.map(
            lambda leaf, reference: jnp.take(leaf, slot, axis=0).astype(
                reference.dtype
            ),
            pool,
            like,
        )

    def write(self, pool: PyTree, slot, params: PyTree, when=None):
        def put(leaf, value):
            value = value.astype(leaf.dtype)
            if when is None:
                return leaf.at[slot].set(value)
            return leaf.at[slot].set(
                jnp.where(when, value, jnp.take(leaf, slot, axis=0))
            )

        return jax.tree.map(put, pool, params)

    def opponents(self, state: PSROState) -> PyTree:
        drawn = [
            self.read(state.pool, learner.opponent) for learner in state.learners
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

    def report(self, index: int, meta: Meta, score, count, admitted) -> None:
        lox.log(
            {
                f"learner_{index}/episode_return": score / jnp.maximum(count, 1.0),
                f"learner_{index}/episodes": count,
                f"learner_{index}/admitted": admitted.astype(jnp.float32),
            }
        )
        if self.pool:
            reference = winrate(meta)[
                len(self.learners) : len(self.learners) + len(self.pool)
            ]
            lox.log({f"learner_{index}/reference": reference.mean()})

    def init(self, key: Key, state) -> PSROState:
        roster: RosterState = state.algorithm_state
        parts = len(self.learners)
        width = state.timestep.terminated.shape[0] // parts
        learners = tuple(
            LearnerState(
                opponent=jnp.array(index),
                returns=jnp.zeros((width,)),
                resets=jnp.array(False),
            )
            for index in range(parts)
        )
        stacked = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *(algorithm_state.params for algorithm_state in roster.algorithm_states),
        )
        pool = jax.tree.map(
            lambda leaf: jnp.zeros((self.capacity, *leaf.shape[1:]), self.dtype)
            .at[:parts]
            .set(leaf.astype(self.dtype)),
            stacked,
        )
        pool = reduce(
            lambda pool, entry: self.write(pool, parts + entry[0], entry[1].params),
            enumerate(self.pool),
            pool,
        )
        occupied = parts + len(self.pool)
        lineages = jnp.array(
            [self.names.index(learner.lineage) for learner in self.learners]
            + [self.names.index(member.lineage) for member in self.pool]
        )
        return PSROState(
            learners=learners,
            pool=pool,
            payoff=jnp.zeros((self.capacity, self.capacity)),
            counts=jnp.zeros((self.capacity, self.capacity)),
            members=jnp.zeros(self.capacity).at[:occupied].set(1.0),
            lineages=jnp.full(self.capacity, -1).at[:occupied].set(lineages),
            cursor=jnp.array(occupied),
            iteration=jnp.array(0),
        )

    def update(
        self,
        state: PSROState,
        key: Key,
        transitions: Transition,
        algorithm_state: RosterState,
    ) -> PSROState:
        following = []
        for index, learner in enumerate(self.learners):
            state, params = self.iterate(
                state,
                index,
                learner,
                transitions,
                algorithm_state,
                jax.random.fold_in(key, index),
            )
            following.append(params)
        pool = reduce(
            lambda pool, entry: self.write(pool, entry[0], entry[1]),
            enumerate(following),
            state.pool,
        )
        return state.replace(pool=pool, iteration=state.iteration + 1)

    def iterate(
        self,
        state: PSROState,
        index: int,
        learner: Learner,
        transitions: Transition,
        algorithm_state: RosterState,
        key: Key,
    ) -> tuple[PSROState, PyTree]:
        sample_key, restart_key = jax.random.split(key)
        learner_state = state.learners[index]
        view = take(transitions, index, len(self.learners), axis=1)

        returns, score, count = self.score(learner_state, view)
        rival = learner_state.opponent
        state = state.replace(
            payoff=self.tally(state.payoff, index, rival, score, -1.0),
            counts=self.tally(state.counts, index, rival, count, 1.0),
        )

        params = algorithm_state.algorithm_states[index].params
        admitted = learner.admit(self.meta(state, index)) & (
            state.cursor < self.capacity
        )
        pool = self.write(state.pool, state.cursor, params, when=admitted)
        resets = admitted & learner.resets
        if learner.resets:
            restart = algorithm_state.initial_states[index].params
            if learner.restart:
                slot = jax.random.categorical(
                    restart_key, jnp.log(learner.restart(self.meta(state, index)))
                )
                restart = self.read(pool, slot, like=restart)
            params = conditional_update(restart, params, admitted)

        self.report(index, self.meta(state, index), score, count, admitted)
        state = state.replace(
            pool=pool,
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
        learners = list(state.learners)
        learners[index] = learner_state.replace(
            returns=returns,
            resets=resets,
            opponent=jax.random.categorical(
                sample_key, jnp.log(learner.solver(self.meta(state, index)))
            ),
        )
        return state.replace(learners=tuple(learners)), params

    def restart(self, state: PSROState, roster: RosterState) -> RosterState:
        for index, learner in enumerate(self.learners):
            if not learner.resets:
                continue
            params = None
            if learner.restart:
                params = self.read(
                    state.pool, index, like=roster.initial_states[index].params
                )
            roster = self.roster.restart(
                roster, index, state.learners[index].resets, params
            )
        return roster

    def __call__(self, key, state):
        opponents = self.opponents(state.pit_state)

        def seat(node):
            if isinstance(node, OpponentState):
                return self.opponent.update(node, key, opponents=opponents)
            return node

        seated = jax.tree.map(
            seat,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, OpponentState),
        )
        return self.dress(
            state.replace(
                algorithm_state=self.restart(state.pit_state, state.algorithm_state),
                environment_state=seated,
            )
        )


def psro(
    algorithm: Algorithm,
    environment: Environment,
    learners,
    play,
    initialize,
    capacity,
    decay,
    load=lambda checkpoint: checkpoint,
    pool=(),
    deal=lambda environment, learners, lineages, **table: lambda state: state,
    **table,
) -> tuple[Roster, Environment, PSRO, Lap | None]:
    learners = list(learners)
    roster = Roster(
        algorithm,
        count=len(learners),
        warmstarts={
            index: load(learner.warmstart)
            for index, learner in enumerate(learners)
            if learner.warmstart is not None
        },
        kl_coefficients=tuple(learner.kl_coefficient for learner in learners),
    )
    environment = Opponent(environment, play, initialize, groups=len(learners))
    pit = PSRO(
        roster=roster,
        opponent=environment,
        learners=learners,
        capacity=capacity,
        decay=decay,
        pool=tuple(
            Member(lineage=spec.lineage, params=load(spec.checkpoint))
            for spec in pool
        ),
    )
    pit = replace(pit, dress=deal(environment, learners, pit.names, **table))
    return roster, environment, pit, None
