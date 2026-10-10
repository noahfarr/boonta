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


def wipe(evicted_slots: Array, archive_size: int) -> Array:
    evicted_slots = jnp.reshape(evicted_slots, (-1,))
    scatter_rows = jnp.where(evicted_slots >= 0, evicted_slots, archive_size)
    wiped_mask = jnp.zeros(archive_size + 1, bool).at[scatter_rows].set(True, mode="drop")
    return wiped_mask[:archive_size]


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

    def expected_visits(self, successor: Successor, start_slot: Array) -> Array:
        columns, weights = successor.columns, successor.weights
        archive_size, _ = jnp.shape(weights)
        start_columns = columns[0] if jnp.shape(columns)[0] == 1 else columns[start_slot]
        return jnp.zeros(archive_size + 1).at[start_columns].add(weights[start_slot])[:archive_size]

    def spill(self, successor: Successor, weight: Array) -> Array:
        columns, weights = successor.columns, successor.weights
        archive_size, *_ = jnp.shape(weight)
        if jnp.shape(columns)[0] == 1:
            flow = weight @ weights
            return jnp.zeros(archive_size + 1).at[columns[0]].add(flow)[:archive_size]
        flow = weight[:, None] * weights
        return jnp.zeros(archive_size + 1).at[columns].add(flow)[:archive_size]

    def forget(self, successor: Successor, wiped_mask: Array) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, _ = jnp.shape(weights)
        landed_mask = wiped_mask[jnp.clip(columns, 0, archive_size - 1)] & (columns < archive_size)
        weights = jnp.where(landed_mask, 0.0, weights)
        if jnp.shape(columns)[0] > 1:
            columns = jnp.where(landed_mask | wiped_mask[:, None], archive_size, columns)
        return Successor(columns=columns, weights=jnp.where(wiped_mask[:, None], 0.0, weights))

    def backup(
        self,
        successor: Successor,
        leaving_slots: Array,
        landing_slots: Array,
        onward_mask: Array,
        key: Key,
    ) -> Successor:
        archive_size, _ = jnp.shape(successor.weights)
        scatter_rows = jnp.where(leaving_slots >= 0, leaving_slots, archive_size)
        leaving_counts = jnp.zeros(archive_size).at[scatter_rows].add(1.0, mode="drop")
        step_sizes = self.rate / jnp.maximum(leaving_counts, 1.0)
        if jnp.shape(successor.columns)[0] == 1:
            return self.blend(successor, leaving_slots, landing_slots, onward_mask, step_sizes)
        return self.merge(successor, leaving_slots, landing_slots, onward_mask, key)

    def blend(
        self,
        successor: Successor,
        leaving_slots: Array,
        landing_slots: Array,
        onward_mask: Array,
        step_sizes: Array,
    ) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, _ = jnp.shape(weights)
        target = jax.nn.one_hot(leaving_slots, archive_size) + self.gamma * jnp.where(
            onward_mask[..., None], weights[jnp.clip(landing_slots, 0)], 0.0
        )
        leaving_rows = jnp.clip(leaving_slots, 0)
        scatter_rows = jnp.where(leaving_slots >= 0, leaving_slots, archive_size)
        return Successor(
            columns=columns,
            weights=weights.at[scatter_rows].add(
                step_sizes[leaving_rows][..., None] * (target - weights[leaving_rows]),
                mode="drop",
            ),
        )

    def merge(
        self,
        successor: Successor,
        leaving_slots: Array,
        landing_slots: Array,
        onward_mask: Array,
        key: Key,
    ) -> Successor:
        columns, weights = successor.columns, successor.weights
        archive_size, width = jnp.shape(weights)
        leaving_rows, landing_rows = jnp.clip(leaving_slots, 0), jnp.clip(landing_slots, 0)
        scatter_rows = jnp.where(leaving_slots >= 0, leaving_slots, archive_size)
        write_priorities = jax.random.uniform(key, jnp.shape(leaving_slots))
        highest_priority = (
            jnp.full(archive_size, -1.0).at[scatter_rows].max(write_priorities, mode="drop")
        )
        write_mask = (leaving_slots >= 0) & (write_priorities == highest_priority[leaving_rows])
        own_columns = columns[leaving_rows]
        landing_columns = columns[landing_rows]
        landed_mask = landing_columns == leaving_rows[..., None]
        reach_columns = jnp.concatenate([landing_columns, leaving_rows[..., None]], axis=-1)
        target = jnp.concatenate(
            [
                self.gamma * jnp.where(onward_mask[..., None], weights[landing_rows], 0.0)
                + landed_mask,
                jnp.where(jnp.any(landed_mask, axis=-1), 0.0, 1.0)[..., None],
            ],
            axis=-1,
        )
        real_mask = reach_columns < archive_size
        match_mask = (reach_columns[..., :, None] == own_columns[..., None, :]) & real_mask[
            ..., None
        ]
        kept_weights = (1.0 - self.rate) * weights[leaving_rows] + self.rate * jnp.einsum(
            "...jk,...j->...k", match_mask.astype(target.dtype), target
        )
        added_weights = jnp.where(
            real_mask & ~jnp.any(match_mask, axis=-1), self.rate * target, 0.0
        )
        top_weights, top_indices = jax.lax.top_k(
            jnp.concatenate([kept_weights, added_weights], axis=-1), width
        )
        top_columns = jnp.take_along_axis(
            jnp.concatenate([own_columns, reach_columns], axis=-1), top_indices, axis=-1
        )
        write_rows = jnp.where(write_mask, leaving_slots, archive_size)
        return Successor(
            columns=columns.at[write_rows].set(
                jnp.where(top_weights > 0.0, top_columns, archive_size), mode="drop"
            ),
            weights=weights.at[write_rows].set(jnp.maximum(top_weights, 0.0), mode="drop"),
        )


class SuccessorFootprint:
    def pour(self, successor: SuccessorRepresentation, table: Successor, term: Array) -> Array:
        return successor.pour(table, term)

    def expected_visits(
        self, successor: SuccessorRepresentation, table: Successor, start_slot: Array
    ) -> Array:
        return successor.expected_visits(table, start_slot)

    def spill(self, successor: SuccessorRepresentation, table: Successor, weight: Array) -> Array:
        return successor.spill(table, weight)


class UniformFootprint:
    def pour(self, successor: SuccessorRepresentation, table: Successor, term: Array) -> Array:
        return jnp.full_like(term, jnp.mean(term))

    def expected_visits(
        self, successor: SuccessorRepresentation, table: Successor, start_slot: Array
    ) -> Array:
        archive_size, _ = jnp.shape(table.weights)
        return jnp.full(archive_size, 1.0 / archive_size)

    def spill(self, successor: SuccessorRepresentation, table: Successor, weight: Array) -> Array:
        return jnp.full_like(weight, jnp.mean(weight))


def piled(size: int, bins: Array, value: Array, lanes: int = 256) -> Array:
    lane_ids = jnp.arange(jnp.shape(bins)[0]) % lanes
    blank = jnp.zeros((lanes, size), jnp.float32)
    return jnp.sum(blank.at[lane_ids, bins].add(value, mode="drop"), axis=0)


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
        present_mask = state.action_counts > 0.0
        action_means = jnp.where(
            present_mask, state.advantage_sums / jnp.maximum(state.action_counts, 1e-12), 0.0
        )
        cell_means = jnp.sum(state.advantage_sums, axis=-1) / jnp.maximum(
            jnp.sum(state.action_counts, axis=-1), 1e-12
        )
        action_probs = state.probability_sums / jnp.maximum(state.cell_visits, 1e-12)[:, None]
        spread = action_probs * (action_means - cell_means[:, None]) ** 2
        return jnp.sum(jnp.where(present_mask, spread, 0.0), axis=-1)

    def init(self, archive_size: int, num_actions: int) -> GainState:
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        return GainState(
            action_counts=table,
            advantage_sums=table,
            probability_sums=table,
            cell_visits=jnp.zeros(archive_size, jnp.float32),
            scale=jnp.zeros((), jnp.float32),
        )

    def forget(self, state: GainState, wiped_mask: Array) -> GainState:
        def wipe_cells(leaf):
            if leaf.ndim == 0:
                return leaf
            return jnp.where(wiped_mask if leaf.ndim == 1 else wiped_mask[:, None], 0.0, leaf)

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
        archive_size, num_actions = jnp.shape(state.action_counts)
        cell_slots = first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~truncated
        actions = jnp.clip(second.action[:-1].astype(jnp.int32), 0, num_actions - 1)
        action_probs = jnp.exp(transitions.aux["log_prob"][:-1].astype(jnp.float32))
        pair_bins = jnp.reshape(
            jnp.where(valid_mask, cell_slots * num_actions + actions, archive_size * num_actions),
            (-1,),
        )
        decay = 1.0 - self.rate

        def tally(value):
            summed = piled(
                archive_size * num_actions,
                pair_bins,
                jnp.reshape(jnp.where(valid_mask, value, 0.0), (-1,)),
            )
            return jnp.reshape(summed, (archive_size, num_actions))

        visit_counts = tally(1.0)
        state = state.replace(
            action_counts=decay * state.action_counts + visit_counts,
            advantage_sums=decay * state.advantage_sums + tally(advantages),
            probability_sums=decay * state.probability_sums + tally(action_probs),
            cell_visits=decay * state.cell_visits + jnp.sum(visit_counts, axis=-1),
        )
        logits = transitions.aux.get("logits")
        if logits is None:
            spread = (1.0 - action_probs) ** 2
        else:
            policy_probs = jax.nn.softmax(logits[:-1].astype(jnp.float32), axis=-1)
            chosen_probs = jnp.take_along_axis(policy_probs, actions[..., None], axis=-1)[..., 0]
            spread = 1.0 - 2.0 * chosen_probs + jnp.sum(policy_probs**2, axis=-1)
        measured = self.noise_scale(self(state), cell_slots, valid_mask, advantages, spread)
        smoothed = jnp.where(
            state.scale > 0.0, decay * state.scale + self.rate * measured, measured
        )
        return state.replace(scale=jnp.where(measured > 0.0, smoothed, state.scale))

    def noise_scale(
        self,
        gain: Array,
        cell_slots: Array,
        valid_mask: Array,
        advantages: Array,
        spread: Array,
    ) -> Array:
        sample_power = jnp.sum(jnp.where(valid_mask, advantages**2 * spread, 0.0))
        signal_power = jnp.sum(jnp.where(valid_mask, gain[jnp.clip(cell_slots, 0)], 0.0))
        return jnp.where(
            signal_power > 0.0,
            jnp.maximum(sample_power - signal_power, 0.0) / jnp.maximum(signal_power, 1e-30),
            0.0,
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


def moments(value: Array, over_mask: Array) -> tuple[Array, Array]:
    count = jnp.maximum(jnp.sum(over_mask, dtype=value.dtype), 1.0)
    centre = jnp.sum(jnp.where(over_mask, value, 0.0)) / count
    return centre, jnp.sqrt(jnp.sum(jnp.where(over_mask, (value - centre) ** 2, 0.0)) / count)


def random_argmax(value: Array, mask: Array, key: Key, shape) -> Array:
    highest = jnp.max(jnp.where(mask, value, -jnp.inf))
    return pick(key, jnp.where(mask & (value >= highest), 0.0, -jnp.inf), shape)


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
        leaving_slots, landing_slots = first.info["cell"], second.info["cell"]
        steps, envs = jnp.shape(leaving_slots)
        done = jnp.reshape(second.done, (steps, envs, -1)).all(axis=-1)
        archive_size, *_ = jnp.shape(state.roots)

        def tread(carry, step):
            table, roots = carry
            leaving_slots, landing_slots, done, evicted_slots, start_mask, key = step
            root_rows = jnp.where(start_mask & (leaving_slots >= 0), leaving_slots, archive_size)
            roots = roots.at[root_rows].add(1.0, mode="drop")
            wiped_mask = wipe(evicted_slots, archive_size)
            table = self.successor.forget(table, wiped_mask)
            table = self.successor.backup(
                table, leaving_slots, landing_slots, ~done & (landing_slots >= 0), key
            )
            return (table, jnp.where(wiped_mask, 0.0, roots)), wiped_mask

        (table, roots), wiped_mask = jax.lax.scan(
            tread,
            (state.successor, state.roots),
            (
                leaving_slots,
                landing_slots,
                done,
                second.info["evicted"],
                first.info["start"],
                jax.random.split(key, steps),
            ),
        )
        wiped_mask = jnp.any(wiped_mask, axis=0)
        return state.replace(
            successor=table,
            roots=roots,
            gain=self.gain.update(self.gain.forget(state.gain, wiped_mask), transitions),
        )

    def select(self, state: SelectorState, wrapper, archive_auto_reset_state, key: Key):
        archive_state = archive_auto_reset_state.archive_state
        eligible_mask = wrapper.archive.eligible(archive_state)
        num_placed = share(self.k, wrapper.num_envs)
        occupied_counts = wrapper.occupied(archive_auto_reset_state, num_placed)
        selected_slots = self.allot(state, eligible_mask, occupied_counts, key, num_placed)
        self.watch(state, archive_state.mask, selected_slots)
        self.saturation(state, eligible_mask, occupied_counts, selected_slots)
        return state, selected_slots

    def allot(
        self,
        state: SelectorState,
        eligible_mask: Array,
        occupied_counts: Array | float,
        key: Key,
        count: int,
    ) -> Array:
        worth = self.relevant(state) * self.gain(state.gain)

        def choose(visits, key):
            factor = self.diminishing_factor(state, visits)
            marginal = self.footprint.pour(self.successor, state.successor, worth * factor)
            chosen_slot = random_argmax(marginal, eligible_mask, key, ())
            spread = self.footprint.expected_visits(self.successor, state.successor, chosen_slot)
            return visits + spread, chosen_slot

        _, selected_slots = jax.lax.scan(
            choose, self.committed_visits(state, occupied_counts), jax.random.split(key, count)
        )
        return selected_slots

    def relevant(self, state: SelectorState) -> Array:
        return self.relevance(self.successor, state.successor, state.roots)

    def diminishing_factor(self, state: SelectorState, visits: Array | float) -> Array:
        scale = state.gain.scale
        return jnp.where(scale > 0.0, 1.0 / (1.0 + visits / jnp.maximum(scale, 1e-30)) ** 2, 1.0)

    def committed_visits(self, state: SelectorState, occupied_counts: Array | float) -> Array:
        return self.footprint.spill(
            self.successor, state.successor, jnp.zeros_like(state.roots) + occupied_counts
        )

    def value(self, state: SelectorState, visits: Array | float = 0.0) -> Array:
        term = self.relevant(state) * self.gain(state.gain) * self.diminishing_factor(state, visits)
        return self.footprint.pour(self.successor, state.successor, term)

    def saturation(
        self,
        state: SelectorState,
        eligible_mask: Array,
        occupied_counts: Array | float,
        selected_slots: Array,
    ) -> None:
        archive_size, *_ = jnp.shape(eligible_mask)
        committed = self.committed_visits(state, occupied_counts)
        restart_counts = jnp.zeros(archive_size).at[selected_slots].add(1.0)
        visits = committed + self.footprint.spill(self.successor, state.successor, restart_counts)
        scale = jnp.maximum(state.gain.scale, 1e-30)
        before = jnp.where(eligible_mask, self.value(state, committed), -jnp.inf)
        best, runner_up = jax.lax.top_k(before, 2)[0]
        chosen_slot = jnp.argmax(restart_counts)
        after = self.value(state, visits)[chosen_slot]
        peak_visits = jnp.max(visits)
        lox.log(
            {
                "mu/peak_load": peak_visits / scale,
                "mu/peak_factor": self.diminishing_factor(state, peak_visits),
                "archive/top_ratio": best / jnp.maximum(runner_up, 1e-30),
                "mu/chosen_discount": after / jnp.maximum(before[chosen_slot], 1e-30),
            }
        )

    def watch(self, state: SelectorState, archive_mask: Array, selected_slots: Array) -> None:
        archive_size, *_ = jnp.shape(archive_mask)
        num_cells = jnp.maximum(jnp.sum(archive_mask, dtype=jnp.float32), 1.0)
        weights = jnp.zeros(archive_size).at[selected_slots].add(
            1.0 / max(jnp.shape(selected_slots)[0], 1)
        )
        entropy = -jnp.sum(jnp.where(weights > 0.0, weights * jnp.log(weights), 0.0))
        value = self.value(state)
        gain = self.gain(state.gain)
        relevance = self.relevant(state)
        reach = self.successor.spill(state.successor, state.roots)
        lox.log(
            {
                "archive/span": jnp.sum(reach) / jnp.maximum(jnp.sum(state.roots), 1.0),
                "archive/noise_scale": state.gain.scale,
                "archive/gain": jnp.sum(jnp.where(archive_mask, gain, 0.0)) / num_cells,
                "archive/value": jnp.sum(jnp.where(archive_mask, value, 0.0)) / num_cells,
                "archive/spread": moments(value, archive_mask)[1],
                "archive/argmax": jnp.max(jnp.where(archive_mask, value, 0.0)),
                "mu/gain": jnp.sum(weights * gain),
                "mu/coverage": jnp.sum(archive_mask & (weights > 0.0), dtype=jnp.float32)
                / num_cells,
                "rho/coverage": jnp.sum(archive_mask & (relevance > 0.0), dtype=jnp.float32)
                / num_cells,
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

    def assign(state, transitions):
        key = jax.random.fold_in(
            jax.random.key(seed), state.algorithm_state.step.astype(jnp.uint32)
        )
        archive_auto_reset_state = wrapper.assign(
            locate(state.environment_state), key, transitions
        )
        return state.replace(
            environment_state=plant(state.environment_state, archive_auto_reset_state)
        )

    return algorithm, environment, assign, lambda state: state
