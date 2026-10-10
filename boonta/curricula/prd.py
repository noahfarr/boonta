from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimation
from boonta.environments.wrappers import archive_auto_reset
from boonta.environments.wrappers.archive_auto_reset import (locate, pick,
                                                             plant, share, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, Transition
from boonta.utils.typing import Environment


@struct.dataclass
class Successor:
    columns: Array
    weights: Array


def wipe(evicted: Array, size: int) -> Array:
    gone = jnp.reshape(evicted, (-1,))
    return jnp.zeros(size + 1, bool).at[jnp.where(gone >= 0, gone, size)].set(True, mode="drop")[:size]


@dataclass
class SuccessorRepresentation:
    gamma: float = 1.0
    rate: float = 0.1
    columns: int | None = None

    def init(self, archive_size: int) -> Successor:
        width = archive_size if self.columns is None else max(1, self.columns)
        return Successor(
            columns=jnp.arange(archive_size)[None]
            if width >= archive_size
            else jnp.full((archive_size, width), archive_size),
            weights=jnp.zeros((archive_size, min(width, archive_size))),
        )

    def pour(self, successor: Successor, term: Array) -> Array:
        return jnp.sum(successor.weights * jnp.append(term, 0.0)[successor.columns], axis=-1)

    def expected_visits(self, successor: Successor, start: Array) -> Array:
        columns, weights = successor.columns, successor.weights
        size, _ = jnp.shape(weights)
        row = columns[0] if jnp.shape(columns)[0] == 1 else columns[start]
        return jnp.zeros(size + 1).at[row].add(weights[start])[:size]

    def spill(self, successor: Successor, weight: Array) -> Array:
        columns, weights = successor.columns, successor.weights
        archive_size, *_ = jnp.shape(weight)
        if jnp.shape(columns)[0] == 1:
            flow = weight @ weights
            return jnp.zeros(archive_size + 1).at[columns[0]].add(flow)[:archive_size]
        flow = weight[:, None] * weights
        return jnp.zeros(archive_size + 1).at[columns].add(flow)[:archive_size]

    def forget(self, successor: Successor, wiped: Array) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, _ = jnp.shape(weights)
        landed = wiped[jnp.clip(columns, 0, archive_size - 1)] & (columns < archive_size)
        weights = jnp.where(landed, 0.0, weights)
        if jnp.shape(columns)[0] > 1:
            columns = jnp.where(landed | wiped[:, None], archive_size, columns)
        return Successor(columns=columns, weights=jnp.where(wiped[:, None], 0.0, weights))

    def backup(
        self, successor: Successor, leaving: Array, landing: Array, onward: Array, key: Key
    ) -> Successor:
        archive_size, _ = jnp.shape(successor.weights)
        index = jnp.where(leaving >= 0, leaving, archive_size)
        crowd = jnp.zeros(archive_size).at[index].add(1.0, mode="drop")
        step = self.rate / jnp.maximum(crowd, 1.0)
        if jnp.shape(successor.columns)[0] == 1:
            return self.blend(successor, leaving, landing, onward, step)
        return self.merge(successor, leaving, landing, onward, key)

    def blend(
        self, successor: Successor, leaving: Array, landing: Array, onward: Array, step: Array
    ) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, _ = jnp.shape(weights)
        target = jax.nn.one_hot(leaving, archive_size) + self.gamma * jnp.where(
            onward[..., None], weights[jnp.clip(landing, 0)], 0.0
        )
        here = jnp.clip(leaving, 0)
        index = jnp.where(leaving >= 0, leaving, archive_size)
        return Successor(
            columns=columns,
            weights=weights.at[index].add(
                step[here][..., None] * (target - weights[here]), mode="drop"
            ),
        )

    def merge(
        self, successor: Successor, leaving: Array, landing: Array, onward: Array, key: Key
    ) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, width = jnp.shape(weights)
        here, there = jnp.clip(leaving, 0), jnp.clip(landing, 0)
        index = jnp.where(leaving >= 0, leaving, archive_size)
        priority = jax.random.uniform(key, jnp.shape(leaving))
        writer = jnp.full(archive_size, -1.0).at[index].max(priority, mode="drop")
        chosen = (leaving >= 0) & (priority == writer[here])
        own = columns[here]
        ahead = columns[there]
        landed = ahead == here[..., None]
        reach = jnp.concatenate([ahead, here[..., None]], axis=-1)
        target = jnp.concatenate(
            [
                self.gamma * jnp.where(onward[..., None], weights[there], 0.0) + landed,
                jnp.where(jnp.any(landed, axis=-1), 0.0, 1.0)[..., None],
            ],
            axis=-1,
        )
        real = reach < archive_size
        match = (reach[..., :, None] == own[..., None, :]) & real[..., None]
        kept = (1.0 - self.rate) * weights[here] + self.rate * jnp.einsum(
            "...jk,...j->...k", match.astype(target.dtype), target
        )
        added = jnp.where(real & ~jnp.any(match, axis=-1), self.rate * target, 0.0)
        top, at = jax.lax.top_k(jnp.concatenate([kept, added], axis=-1), width)
        picked = jnp.take_along_axis(jnp.concatenate([own, reach], axis=-1), at, axis=-1)
        row = jnp.where(chosen, leaving, archive_size)
        return Successor(
            columns=columns.at[row].set(jnp.where(top > 0.0, picked, archive_size), mode="drop"),
            weights=weights.at[row].set(jnp.maximum(top, 0.0), mode="drop"),
        )


class SuccessorFootprint:
    def pour(self, successor: SuccessorRepresentation, table: Successor, term: Array) -> Array:
        return successor.pour(table, term)

    def expected_visits(
        self, successor: SuccessorRepresentation, table: Successor, start: Array
    ) -> Array:
        return successor.expected_visits(table, start)

    def spill(self, successor: SuccessorRepresentation, table: Successor, weight: Array) -> Array:
        return successor.spill(table, weight)


class UniformFootprint:
    def pour(self, successor: SuccessorRepresentation, table: Successor, term: Array) -> Array:
        return jnp.full_like(term, jnp.mean(term))

    def expected_visits(
        self, successor: SuccessorRepresentation, table: Successor, start: Array
    ) -> Array:
        size, _ = jnp.shape(table.weights)
        return jnp.full(size, 1.0 / size)

    def spill(self, successor: SuccessorRepresentation, table: Successor, weight: Array) -> Array:
        return jnp.full_like(weight, jnp.mean(weight))


def piled(size: int, index: Array, value: Array, lanes: int = 256) -> Array:
    lane = jnp.arange(jnp.shape(index)[0]) % lanes
    blank = jnp.zeros((lanes, size), jnp.float32)
    return jnp.sum(blank.at[lane, index].add(value, mode="drop"), axis=0)


@struct.dataclass
class GainState:
    action_counts: Array
    advantage_sums: Array
    probability_sums: Array
    cell_visits: Array
    scale: Array


@dataclass
class Advantage:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rate: float = 0.2

    def __call__(self, state: GainState) -> Array:
        present = state.action_counts > 0.0
        means = jnp.where(present, state.advantage_sums / jnp.maximum(state.action_counts, 1e-12), 0.0)
        level = jnp.sum(state.advantage_sums, axis=-1) / jnp.maximum(
            jnp.sum(state.action_counts, axis=-1), 1e-12
        )
        taken = state.probability_sums / jnp.maximum(state.cell_visits, 1e-12)[:, None]
        return jnp.sum(jnp.where(present, taken * (means - level[:, None]) ** 2, 0.0), axis=-1)

    def init(self, archive_size: int, num_actions: int) -> GainState:
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        return GainState(
            action_counts=table,
            advantage_sums=table,
            probability_sums=table,
            cell_visits=jnp.zeros(archive_size, jnp.float32),
            scale=jnp.zeros((), jnp.float32),
        )

    def forget(self, state: GainState, wiped: Array) -> GainState:
        def wipe_cells(leaf):
            if leaf.ndim == 0:
                return leaf
            return jnp.where(wiped if leaf.ndim == 1 else wiped[:, None], 0.0, leaf)

        return jax.tree.map(wipe_cells, state)

    def update(self, state: GainState, transitions: Transition) -> GainState:
        first, second = transitions.first, transitions.second
        values = transitions.aux["value"]
        truncated = second.truncated[:-1].astype(bool)
        advantages, _ = generalized_advantage_estimation(
            jax.tree.map(lambda leaf: leaf[:-1], transitions),
            values[:-1],
            values[-1],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        capacity, actions = jnp.shape(state.action_counts)
        cell = first.info["cell"][:-1]
        valid = (cell >= 0) & ~truncated
        action = jnp.clip(second.action[:-1].astype(jnp.int32), 0, actions - 1)
        taken = jnp.exp(transitions.aux["log_prob"][:-1].astype(jnp.float32))
        pair = jnp.reshape(jnp.where(valid, cell * actions + action, capacity * actions), (-1,))
        keep = 1.0 - self.rate

        def tally(value):
            summed = piled(capacity * actions, pair, jnp.reshape(jnp.where(valid, value, 0.0), (-1,)))
            return jnp.reshape(summed, (capacity, actions))

        seen = tally(1.0)
        state = state.replace(
            action_counts=keep * state.action_counts + seen,
            advantage_sums=keep * state.advantage_sums + tally(advantages),
            probability_sums=keep * state.probability_sums + tally(taken),
            cell_visits=keep * state.cell_visits + jnp.sum(seen, axis=-1),
        )
        logits = transitions.aux.get("logits")
        if logits is None:
            spread = (1.0 - taken) ** 2
        else:
            probs = jax.nn.softmax(logits[:-1].astype(jnp.float32), axis=-1)
            chosen = jnp.take_along_axis(probs, action[..., None], axis=-1)[..., 0]
            spread = 1.0 - 2.0 * chosen + jnp.sum(probs**2, axis=-1)
        fresh = self.noise_scale(self(state), cell, valid, advantages, spread)
        scale = jnp.where(state.scale > 0.0, keep * state.scale + self.rate * fresh, fresh)
        return state.replace(scale=jnp.where(fresh > 0.0, scale, state.scale))

    def noise_scale(
        self, gain: Array, cell: Array, valid: Array, advantages: Array, spread: Array
    ) -> Array:
        sample = jnp.sum(jnp.where(valid, advantages**2 * spread, 0.0))
        signal = jnp.sum(jnp.where(valid, gain[jnp.clip(cell, 0)], 0.0))
        return jnp.where(
            signal > 0.0, jnp.maximum(sample - signal, 0.0) / jnp.maximum(signal, 1e-30), 0.0
        )


@struct.dataclass
class SelectorState:
    successor: Successor
    roots: Array
    gain: GainState


class SuccessorRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, roots: Array
    ) -> Array:
        mass = successor.spill(table, roots)
        return mass / jnp.maximum(jnp.sum(mass), 1e-12)


class UniformRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, roots: Array
    ) -> Array:
        return jnp.ones_like(roots)


def moments(value: Array, over: Array) -> tuple[Array, Array]:
    count = jnp.maximum(jnp.sum(over, dtype=value.dtype), 1.0)
    centre = jnp.sum(jnp.where(over, value, 0.0)) / count
    return centre, jnp.sqrt(jnp.sum(jnp.where(over, (value - centre) ** 2, 0.0)) / count)


def random_argmax(value: Array, mask: Array, key: Key, shape) -> Array:
    top = jnp.max(jnp.where(mask, value, -jnp.inf))
    return pick(key, jnp.where(mask & (value >= top), 0.0, -jnp.inf), shape)


@dataclass
class Selector(archive_auto_reset.Selector):
    successor: SuccessorRepresentation
    footprint: SuccessorFootprint | UniformFootprint
    relevance: SuccessorRelevance | UniformRelevance
    gain: Advantage
    k: float = 4.0

    def init(self, archive_size: int, num_actions: int) -> SelectorState:
        return SelectorState(
            successor=self.successor.init(archive_size),
            roots=jnp.zeros(archive_size),
            gain=self.gain.init(archive_size, num_actions),
        )

    def update(self, state: SelectorState, key: Key, transitions: Transition) -> SelectorState:
        first, second = transitions.first, transitions.second
        leaving, landing = first.info["cell"], second.info["cell"]
        steps, envs = jnp.shape(leaving)
        done = jnp.reshape(second.done, (steps, envs, -1)).all(axis=-1)
        size, *_ = jnp.shape(state.roots)

        def tread(held, rung):
            table, roots = held
            leaving, landing, done, evicted, start, key = rung
            roots = roots.at[jnp.where(start & (leaving >= 0), leaving, size)].add(1.0, mode="drop")
            wiped = wipe(evicted, size)
            table = self.successor.forget(table, wiped)
            table = self.successor.backup(table, leaving, landing, ~done & (landing >= 0), key)
            return (table, jnp.where(wiped, 0.0, roots)), wiped

        (table, roots), wiped = jax.lax.scan(
            tread,
            (state.successor, state.roots),
            (
                leaving,
                landing,
                done,
                second.info["evicted"],
                first.info["start"],
                jax.random.split(key, steps),
            ),
        )
        wiped = jnp.any(wiped, axis=0)
        return state.replace(
            successor=table,
            roots=roots,
            gain=self.gain.update(self.gain.forget(state.gain, wiped), transitions),
        )

    def select(self, state: SelectorState, wrapper, archive_auto_reset_state, key: Key):
        archive_state = archive_auto_reset_state.archive_state
        mask = wrapper.archive.eligible(archive_state)
        placed = share(self.k, wrapper.num_envs)
        occupied = wrapper.occupied(archive_auto_reset_state, placed)
        index = self.allot(state, mask, occupied, key, placed)
        self.watch(state, archive_state.mask, index)
        self.saturation(state, mask, occupied, index)
        return state, index

    def allot(
        self, state: SelectorState, mask: Array, occupied: Array | float, key: Key, count: int
    ) -> Array:
        worth = self.relevant(state) * self.gain(state.gain)

        def assign(visits, key):
            factor = self.diminishing_factor(state, visits)
            marginal = self.footprint.pour(self.successor, state.successor, worth * factor)
            cell = random_argmax(marginal, mask, key, ())
            spread = self.footprint.expected_visits(self.successor, state.successor, cell)
            return visits + spread, cell

        _, index = jax.lax.scan(
            assign, self.committed_visits(state, occupied), jax.random.split(key, count)
        )
        return index

    def relevant(self, state: SelectorState) -> Array:
        return self.relevance(self.successor, state.successor, state.roots)

    def diminishing_factor(self, state: SelectorState, visits: Array | float) -> Array:
        scale = state.gain.scale
        return jnp.where(scale > 0.0, 1.0 / (1.0 + visits / jnp.maximum(scale, 1e-30)) ** 2, 1.0)

    def committed_visits(self, state: SelectorState, occupied: Array | float) -> Array:
        return self.footprint.spill(
            self.successor, state.successor, jnp.zeros_like(state.roots) + occupied
        )

    def value(self, state: SelectorState, visits: Array | float = 0.0) -> Array:
        term = self.relevant(state) * self.gain(state.gain) * self.diminishing_factor(state, visits)
        return self.footprint.pour(self.successor, state.successor, term)

    def saturation(
        self, state: SelectorState, mask: Array, occupied: Array | float, index: Array
    ) -> None:
        size, *_ = jnp.shape(mask)
        committed = self.committed_visits(state, occupied)
        restarts = jnp.zeros(size).at[index].add(1.0)
        visits = committed + self.footprint.spill(self.successor, state.successor, restarts)
        scale = jnp.maximum(state.gain.scale, 1e-30)
        before = jnp.where(mask, self.value(state, committed), -jnp.inf)
        best, runner_up = jax.lax.top_k(before, 2)[0]
        chosen = jnp.argmax(restarts)
        after = self.value(state, visits)[chosen]
        peak = jnp.max(visits)
        lox.log(
            {
                "mu/peak_load": peak / scale,
                "mu/peak_factor": self.diminishing_factor(state, peak),
                "archive/top_ratio": best / jnp.maximum(runner_up, 1e-30),
                "mu/chosen_discount": after / jnp.maximum(before[chosen], 1e-30),
            }
        )

    def watch(self, state: SelectorState, mask: Array, index: Array) -> None:
        size, *_ = jnp.shape(mask)
        count = jnp.maximum(jnp.sum(mask, dtype=jnp.float32), 1.0)
        weights = jnp.zeros(size).at[index].add(1.0 / max(jnp.shape(index)[0], 1))
        entropy = -jnp.sum(jnp.where(weights > 0.0, weights * jnp.log(weights), 0.0))
        value = self.value(state)
        gain = self.gain(state.gain)
        relevance = self.relevant(state)
        reach = self.successor.spill(state.successor, state.roots)
        lox.log(
            {
                "archive/span": jnp.sum(reach) / jnp.maximum(jnp.sum(state.roots), 1.0),
                "archive/noise_scale": state.gain.scale,
                "archive/gain": jnp.sum(jnp.where(mask, gain, 0.0)) / count,
                "archive/value": jnp.sum(jnp.where(mask, value, 0.0)) / count,
                "archive/spread": moments(value, mask)[1],
                "archive/argmax": jnp.max(jnp.where(mask, value, 0.0)),
                "mu/gain": jnp.sum(weights * gain),
                "mu/coverage": jnp.sum(mask & (weights > 0.0), dtype=jnp.float32) / count,
                "rho/coverage": jnp.sum(mask & (relevance > 0.0), dtype=jnp.float32) / count,
                "mu/entropy": entropy,
                "mu/support": jnp.exp(entropy),
                "mu/concentration": jnp.max(weights),
                "mu/rho": jnp.sum(jnp.where(state.roots > 0.0, weights, 0.0)),
            }
        )


def pit(
    algorithm: Algorithm, environment: Environment, seed: int = 0, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    wrapper = unwrap(environment)

    def place(state, transitions):
        key = jax.random.fold_in(
            jax.random.key(seed), state.algorithm_state.step.astype(jnp.uint32)
        )
        placed = wrapper.place(locate(state.environment_state), key, transitions)
        return state.replace(environment_state=plant(state.environment_state, placed))

    return algorithm, environment, place, lambda state: state
