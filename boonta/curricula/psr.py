from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
import flax.linen as nn
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Transition
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
    writers: int | None = None

    def init(self, archive_size: int) -> Successor:
        width = archive_size if self.columns is None else max(1, self.columns)
        return Successor(
            columns=jnp.arange(archive_size)[None]
            if width >= archive_size
            else jnp.full((archive_size, width), archive_size),
            weights=jnp.zeros((archive_size, min(width, archive_size))),
        )

    def pour(self, successor: Successor, term: Array) -> Array:
        size, *_ = jnp.shape(term)
        gathered = jnp.where(
            successor.columns < size, term[jnp.clip(successor.columns, 0, size - 1)], 0.0
        )
        return jnp.sum(successor.weights * gathered, axis=-1)

    def pour_squared(self, successor: Successor, term: Array) -> Array:
        size, *_ = jnp.shape(term)
        gathered = jnp.where(
            successor.columns < size, term[jnp.clip(successor.columns, 0, size - 1)], 0.0
        )
        return jnp.sum(successor.weights**2 * gathered, axis=-1)

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
        if self.writers is not None and self.writers < jnp.shape(leaving_slots)[0]:
            # Only a cell's writer keeps its row, so the rest of the work runs on the writers alone. Beyond
            # `writers` of them, a random subset skips this step.
            _, writing = jax.lax.top_k(
                jnp.where(write_mask, 1.0 + write_priorities, 0.0), self.writers
            )
            leaving_slots, landing_slots, onward_mask, write_mask = (
                leaf[writing] for leaf in (leaving_slots, landing_slots, onward_mask, write_mask)
            )
            leaving_rows, landing_rows = jnp.clip(leaving_slots, 0), jnp.clip(landing_slots, 0)
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

    def pour_squared(
        self, successor: SuccessorRepresentation, table: Successor, term: Array
    ) -> Array:
        return successor.pour_squared(table, term)

    def expected_visits(
        self, successor: SuccessorRepresentation, table: Successor, start_slot: Array
    ) -> Array:
        return successor.expected_visits(table, start_slot)

    def spill(self, successor: SuccessorRepresentation, table: Successor, weight: Array) -> Array:
        return successor.spill(table, weight)


class UniformFootprint:
    def pour(self, successor: SuccessorRepresentation, table: Successor, term: Array) -> Array:
        return jnp.full_like(term, jnp.mean(term))

    def pour_squared(
        self, successor: SuccessorRepresentation, table: Successor, term: Array
    ) -> Array:
        archive_size, *_ = jnp.shape(term)
        return jnp.full_like(term, jnp.mean(term) / archive_size)

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
    square_sums: Array
    probability_sums: Array
    cell_visits: Array
    scale: Array


def signal(state: GainState) -> Array:
    present_mask = state.action_counts > 0.0
    visits = jnp.sum(state.action_counts, axis=-1)
    groups = jnp.sum(present_mask, axis=-1)
    action_means = jnp.where(
        present_mask, state.advantage_sums / jnp.maximum(state.action_counts, 1e-12), 0.0
    )
    cell_means = jnp.sum(state.advantage_sums, axis=-1) / jnp.maximum(visits, 1e-12)
    between = jnp.sum(
        jnp.where(present_mask, state.action_counts * (action_means - cell_means[:, None]) ** 2, 0.0),
        axis=-1,
    )
    within = jnp.sum(
        jnp.where(
            present_mask,
            state.square_sums - state.advantage_sums**2 / jnp.maximum(state.action_counts, 1e-12),
            0.0,
        ),
        axis=-1,
    )
    noise = within / jnp.maximum(visits - groups, 1.0)
    return jnp.maximum(between - jnp.maximum(groups - 1, 0) * noise, 0.0) / jnp.maximum(visits, 1e-12)


@dataclass
class Advantage:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rate: float = 0.2

    def __call__(self, state: GainState) -> Array:
        return self.power(state)

    def power(self, state: GainState) -> Array:
        present_mask = state.action_counts > 0.0
        action_probs = state.probability_sums / jnp.maximum(state.cell_visits, 1e-12)[:, None]
        spread = action_probs * self.centre(state) ** 2
        return jnp.sum(jnp.where(present_mask, spread, 0.0), axis=-1)

    def centre(self, state: GainState) -> Array:
        present_mask = state.action_counts > 0.0
        action_means = jnp.where(
            present_mask, state.advantage_sums / jnp.maximum(state.action_counts, 1e-12), 0.0
        )
        cell_means = jnp.sum(state.advantage_sums, axis=-1) / jnp.maximum(
            jnp.sum(state.action_counts, axis=-1), 1e-12
        )
        return action_means - cell_means[:, None]

    def init(self, archive_size: int, num_actions: int) -> GainState:
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        return GainState(
            action_counts=table,
            advantage_sums=table,
            square_sums=table,
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
        advantages, _ = generalized_advantage_estimator(
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
            square_sums=decay * state.square_sums + tally(advantages**2),
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
        measured = self.noise_scale(self.power(state), cell_slots, valid_mask, advantages, spread)
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


@dataclass
class SignalGain(Advantage):
    """The debiased between-action spread of a cell's advantages, signal(state), as the gain: the
    variance of the per-action means with the noise that the within-action scatter predicts taken
    out, clipped at zero. Everything else as the advantage gain."""

    def __call__(self, state: GainState) -> Array:
        return jnp.maximum(signal(state), 0.0)


@dataclass
class GreedyAdvantage(Advantage):
    shrinkage: float = 4.0

    def __call__(self, state: GainState) -> Array:
        present_mask = state.action_counts > 0.0
        trust = state.action_counts / (state.action_counts + self.shrinkage)
        return jnp.max(jnp.where(present_mask, trust * self.centre(state), 0.0), axis=-1)


@dataclass
class UniformGain(Advantage):
    def __call__(self, state: GainState) -> Array:
        return jnp.ones_like(state.cell_visits)


@struct.dataclass
class TargetState(GainState):
    target_sums: Array
    target: PyTree


@dataclass
class TargetAdvantage(Advantage):
    """The learner's advantage for the actions an already trained policy would take.

    By the performance-difference lemma, J(target) - J(learner) is the expectation, over the
    target's visitation, of sum_a target(a|x) A(x, a). Paired with TargetRelevance, a cell scores
    its share of that gap. The target's action probabilities are measured on the states the
    learner visits in each cell and averaged per cell like the learner's own. A cell where the
    learner's own choice beats the target's has a negative advantage there and gains nothing.
    Actions the learner never took in a cell have no advantage estimate and count as zero.

    The checkpoint must come from the same recipe, so its parameters fit the learner's network.
    The target is fed the learner's recurrent carry, so the measurement is exact only for
    feedforward policies.
    """

    checkpoint: str | None = None
    seed: int = 0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        assert self.checkpoint is not None, (
            "TargetAdvantage needs a checkpoint: use relevance=target or set gain.checkpoint"
        )
        self.algorithm = algorithm
        self.target, *_ = restore_policy(
            algorithm, environment, self.checkpoint, jax.random.key(self.seed)
        )

    def __call__(self, state: TargetState) -> Array:
        present_mask = state.action_counts > 0.0
        target_probs = state.target_sums / jnp.maximum(state.cell_visits, 1e-12)[:, None]
        advantage = jnp.sum(jnp.where(present_mask, target_probs * self.centre(state), 0.0), axis=-1)
        return jnp.maximum(advantage, 0.0)

    def init(self, archive_size: int, num_actions: int) -> TargetState:
        assert hasattr(self, "target"), "TargetAdvantage needs prepare before it is used"
        state = super().init(archive_size, num_actions)
        return TargetState(
            **vars(state),
            target_sums=jnp.zeros((archive_size, num_actions), jnp.float32),
            target=self.target,
        )

    def forget(self, state: TargetState, wiped_mask: Array) -> TargetState:
        target = state.target
        return super().forget(state.replace(target=None), wiped_mask).replace(target=target)

    def update(self, state: TargetState, transitions: Transition) -> TargetState:
        state = super().update(state, transitions)
        first = transitions.first
        _, _, _, aux = self.algorithm.step(
            state.target, transitions.carry, jax.random.key(self.seed), first, 1.0
        )
        assert aux.get("logits") is not None, "TargetAdvantage needs a policy with logits"
        archive_size, num_actions = jnp.shape(state.action_counts)
        target_probs = jax.nn.softmax(aux["logits"][:-1].astype(jnp.float32), axis=-1)
        cell_slots = first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~transitions.second.truncated[:-1].astype(bool)
        rows = jnp.reshape(jnp.where(valid_mask, cell_slots, archive_size), (-1,))
        tallied = jnp.zeros((archive_size, num_actions)).at[rows].add(
            jnp.reshape(target_probs, (-1, num_actions)), mode="drop"
        )
        return state.replace(target_sums=(1.0 - self.rate) * state.target_sums + tallied)


@struct.dataclass
class HalvedState(TargetState):
    half_counts: Array
    half_sums: Array


def ranks(value: Array, mask: Array) -> Array:
    """Ranks of `value` among the masked entries, ties sharing their mean rank; 0 elsewhere."""
    keyed = jnp.where(mask, value, jnp.inf)
    ordered = jnp.sort(keyed)
    low = jnp.searchsorted(ordered, keyed, side="left")
    high = jnp.searchsorted(ordered, keyed, side="right")
    return jnp.where(mask, (low + high - 1) / 2.0, 0.0)


def rank_correlation(first: Array, second: Array, mask: Array) -> Array:
    """Spearman's correlation of two scores over the masked cells."""
    count = jnp.maximum(jnp.sum(mask, dtype=jnp.float32), 1.0)

    def centred(value):
        rank = ranks(value, mask)
        return jnp.where(mask, rank - jnp.sum(rank) / count, 0.0)

    a, b = centred(first), centred(second)
    norm = jnp.sqrt(jnp.sum(a**2) * jnp.sum(b**2))
    return jnp.where(norm > 0.0, jnp.sum(a * b) / jnp.maximum(norm, 1e-30), 0.0)


@dataclass
class DoubleGreedyAdvantage(TargetAdvantage):
    """The greedy gain, max_a A(x, a), without the upward bias of taking a maximum of noisy means.

    Environments are split by parity into two halves that tally their advantages apart. Each half
    picks the action whose shrunk centred mean is largest, the other half scores that action, and
    the two directions are averaged and clipped at zero, as in double Q-learning. The halves are
    independent rollouts, so noise that makes an action win in one half does not follow it into
    the other.

    With a checkpoint, the target's gain is measured alongside without steering anything, and
    `screen` logs the rank correlations of the target, single greedy and double greedy gains over
    the archive's cells, and over the cells with at least `screen_visits` (decayed) visits.
    """

    shrinkage: float = 4.0
    screen_visits: float = 10.0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        if self.checkpoint is not None:
            super().prepare(algorithm, environment)

    def init(self, archive_size: int, num_actions: int) -> HalvedState:
        state = Advantage.init(self, archive_size, num_actions)
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        halves = jnp.zeros((2, archive_size, num_actions), jnp.float32)
        return HalvedState(
            **vars(state),
            target_sums=table,
            target=getattr(self, "target", None),
            half_counts=halves,
            half_sums=halves,
        )

    def update(self, state: HalvedState, transitions: Transition) -> HalvedState:
        if self.checkpoint is not None:
            state = super().update(state, transitions)
        else:
            state = Advantage.update(self, state, transitions)
        first, second = transitions.first, transitions.second
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimator(
            jax.tree.map(lambda leaf: leaf[:-1], transitions),
            values[:-1],
            values[-1],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        _, archive_size, num_actions = jnp.shape(state.half_counts)
        cell_slots = first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~second.truncated[:-1].astype(bool)
        actions = jnp.clip(second.action[:-1].astype(jnp.int32), 0, num_actions - 1)
        half = jnp.broadcast_to(jnp.arange(cell_slots.shape[-1]) % 2, cell_slots.shape)
        size = 2 * archive_size * num_actions
        bins = jnp.reshape(
            jnp.where(
                valid_mask,
                (half * archive_size + cell_slots) * num_actions + actions,
                size,
            ),
            (-1,),
        )

        def tally(value):
            summed = piled(size, bins, jnp.reshape(jnp.where(valid_mask, value, 0.0), (-1,)))
            return jnp.reshape(summed, (2, archive_size, num_actions))

        decay = 1.0 - self.rate
        return state.replace(
            half_counts=decay * state.half_counts + tally(1.0),
            half_sums=decay * state.half_sums + tally(advantages),
        )

    def greedy(self, state: GainState) -> Array:
        return GreedyAdvantage.__call__(self, state)

    def __call__(self, state: HalvedState) -> Array:
        counts, sums = state.half_counts, state.half_sums
        present_mask = counts > 0.0
        means = jnp.where(present_mask, sums / jnp.maximum(counts, 1e-12), 0.0)
        cell_means = jnp.sum(sums, axis=-1) / jnp.maximum(jnp.sum(counts, axis=-1), 1e-12)
        shrunk = jnp.where(
            present_mask, counts / (counts + self.shrinkage) * (means - cell_means[..., None]), 0.0
        )
        picked = jnp.argmax(jnp.where(present_mask, shrunk, -jnp.inf), axis=-1)
        scored = jnp.take_along_axis(shrunk[::-1], picked[..., None], axis=-1)[..., 0]
        scored = jnp.where(jnp.any(present_mask, axis=-1), scored, 0.0)
        return jnp.maximum(jnp.mean(scored, axis=0), 0.0)

    def screen(self, state: HalvedState, archive_mask: Array) -> None:
        well_mask = archive_mask & (state.cell_visits >= self.screen_visits)
        single, double = self.greedy(state), self(state)
        gains = {"single": single, "double": double}
        if self.checkpoint is not None:
            gains["target"] = TargetAdvantage.__call__(self, state)
        names = list(gains)
        metrics = {
            "screen/well_visited": jnp.sum(well_mask, dtype=jnp.float32)
            / jnp.maximum(jnp.sum(archive_mask, dtype=jnp.float32), 1.0)
        }
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                metrics[f"screen/{a}_{b}"] = rank_correlation(gains[a], gains[b], archive_mask)
                metrics[f"screen/{a}_{b}_well"] = rank_correlation(gains[a], gains[b], well_mask)
        for name, gain in gains.items():
            metrics[f"screen/{name}_mean"] = jnp.sum(jnp.where(well_mask, gain, 0.0)) / jnp.maximum(
                jnp.sum(well_mask, dtype=jnp.float32), 1.0
            )
        lox.log(metrics)


@struct.dataclass
class ModelState(GainState):
    tries: Array
    payoff_sums: Array
    discount_sums: Array
    successor_slots: Array
    value_sums: Array
    value_counts: Array
    policy_sums: Array


@dataclass
class ModelAdvantage(Advantage):
    def power(self, state: ModelState) -> Array:
        tried_mask = state.tries > 0.0
        spread = self.policy(state) ** 2 * self.centre(state) ** 2
        return jnp.sum(jnp.where(tried_mask, spread, 0.0), axis=-1)

    def policy(self, state: ModelState) -> Array:
        return state.policy_sums / jnp.maximum(
            jnp.sum(state.policy_sums, axis=-1, keepdims=True), 1e-12
        )

    def centre(self, state: ModelState) -> Array:
        tried_mask = state.tries > 0.0
        values = state.value_sums / jnp.maximum(state.value_counts, 1e-12)
        successor_values = jnp.where(
            state.successor_slots >= 0, values[jnp.clip(state.successor_slots, 0)], 0.0
        )
        payoffs = state.payoff_sums / jnp.maximum(state.tries, 1.0)
        discounts = state.discount_sums / jnp.maximum(state.tries, 1.0)
        advantages = payoffs + discounts * successor_values - values[:, None]
        baseline = jnp.sum(
            jnp.where(tried_mask, self.policy(state) * advantages, 0.0), axis=-1, keepdims=True
        )
        return jnp.where(tried_mask, advantages - baseline, 0.0)

    def init(self, archive_size: int, num_actions: int) -> ModelState:
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        column = jnp.zeros(archive_size, jnp.float32)
        return ModelState(
            action_counts=table,
            advantage_sums=table,
            square_sums=table,
            probability_sums=table,
            cell_visits=column,
            scale=jnp.zeros((), jnp.float32),
            tries=table,
            payoff_sums=table,
            discount_sums=table,
            successor_slots=jnp.full((archive_size, num_actions), -1, jnp.int32),
            value_sums=column,
            value_counts=column,
            policy_sums=table,
        )

    def forget(self, state: ModelState, wiped_mask: Array) -> ModelState:
        lost_mask = (state.successor_slots >= 0) & wiped_mask[jnp.clip(state.successor_slots, 0)]
        successor_slots = jnp.where(wiped_mask[:, None] | lost_mask, -1, state.successor_slots)
        return super().forget(state, wiped_mask).replace(successor_slots=successor_slots)

    def exit(self, transitions: Transition) -> tuple[Array, Array, Array, Array]:
        first, second = transitions.first, transitions.second
        leaving_slots, landing_slots = first.info["cell"], second.info["cell"]
        ending_mask = second.terminated.astype(bool)
        truncation_mask = second.truncated.astype(bool)
        closed_mask = ending_mask | truncation_mask | (landing_slots != leaving_slots)
        rewards = second.reward.astype(jnp.float32)
        _, envs = jnp.shape(leaving_slots)

        def back(later, step):
            payoff, discount, successor_slots, known_mask = later
            reward, closed, ending, truncation, landing = step
            payoff = jnp.where(closed, reward, reward + self.gamma * payoff)
            discount = jnp.where(
                closed, self.gamma * (1.0 - ending.astype(jnp.float32)), self.gamma * discount
            )
            successor_slots = jnp.where(closed, jnp.where(ending, -1, landing), successor_slots)
            known_mask = jnp.where(closed, ~truncation & (ending | (landing >= 0)), known_mask)
            exited = (payoff, discount, successor_slots, known_mask)
            return exited, exited

        start = (
            jnp.zeros(envs),
            jnp.zeros(envs),
            jnp.full(envs, -1, jnp.int32),
            jnp.zeros(envs, bool),
        )
        _, exited = jax.lax.scan(
            back,
            start,
            (rewards, closed_mask, ending_mask, truncation_mask, landing_slots),
            reverse=True,
        )
        return exited

    def update(self, state: ModelState, transitions: Transition) -> ModelState:
        state = super().update(state, transitions)
        archive_size, num_actions = jnp.shape(state.tries)
        payoffs, discounts, successor_slots, known_mask = jax.tree.map(
            lambda leaf: jnp.reshape(leaf, (-1,)), self.exit(transitions)
        )
        leaving_slots = jnp.reshape(transitions.first.info["cell"], (-1,))
        actions = jnp.clip(
            jnp.reshape(transitions.second.action, (-1,)).astype(jnp.int32), 0, num_actions - 1
        )
        known_mask = known_mask & (leaving_slots >= 0)
        rows = jnp.where(known_mask, leaving_slots, archive_size)
        surviving_rows = jnp.where(known_mask & (successor_slots >= 0), leaving_slots, archive_size)
        visited_rows = jnp.where(leaving_slots >= 0, leaving_slots, archive_size)
        values = jnp.reshape(transitions.aux["value"], (-1,)).astype(jnp.float32)
        visit_probs = jax.nn.softmax(
            jnp.reshape(transitions.aux["logits"], (-1, num_actions)).astype(jnp.float32), axis=-1
        )
        decay = 1.0 - self.rate
        return state.replace(
            tries=state.tries.at[rows, actions].add(1.0, mode="drop"),
            payoff_sums=state.payoff_sums.at[rows, actions].add(payoffs, mode="drop"),
            discount_sums=state.discount_sums.at[rows, actions].add(discounts, mode="drop"),
            successor_slots=state.successor_slots.at[surviving_rows, actions].set(
                successor_slots, mode="drop"
            ),
            value_sums=decay * state.value_sums
            + jnp.zeros(archive_size).at[visited_rows].add(values, mode="drop"),
            value_counts=decay * state.value_counts
            + jnp.zeros(archive_size).at[visited_rows].add(1.0, mode="drop"),
            policy_sums=decay * state.policy_sums
            + jnp.zeros((archive_size, num_actions)).at[visited_rows].add(visit_probs, mode="drop"),
        )


@struct.dataclass
class SelectorState:
    successor: Successor
    reach: Array
    gain: GainState
    direction: PyTree = None
    # The archive's cell keys and occupancy, filled in by select for relevances that need them.
    keys: Array = None
    mask: Array = None
    # The side environments a search relevance plays with, when it has them.
    played: PyTree = None


class CountedRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, reach: Array, **cells
    ) -> Array:
        return reach / jnp.maximum(jnp.sum(reach), 1e-12)


class UniformRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, reach: Array, **cells
    ) -> Array:
        return jnp.ones_like(reach)


def fold(keys: Array) -> Array:
    """Folds each row of cell keys into one unsigned integer so keys can be sorted and searched."""
    keys = jnp.asarray(keys).astype(jnp.uint32)
    folded = jnp.zeros(keys.shape[:-1], jnp.uint32)
    for column in range(keys.shape[-1]):
        folded = folded * jnp.uint32(1000003) + keys[..., column]
    return folded


def archive_auto_reset_state(environment_state: PyTree) -> ArchiveAutoResetState:
    found = []
    jax.tree.map(
        lambda node: found.append(node) if isinstance(node, ArchiveAutoResetState) else None,
        environment_state,
        is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
    )
    return found[0]


def restore_policy(algorithm: Algorithm, environment: Environment, checkpoint: str, key: Key):
    """Loads an already trained policy's algorithm state into the learner's algorithm."""
    from boonta.utils.checkpoint import resolve_checkpoint, restore_item

    environment_state, timestep = environment.init(key)
    algorithm_state, carry = algorithm.init(key, timestep)
    path = Path(resolve_checkpoint(checkpoint)) / "algorithm_state"
    algorithm_state = restore_item(str(path.absolute()), algorithm_state)
    return algorithm_state, environment_state, carry, timestep


@dataclass
class TargetRelevance:
    """The visitation of an already trained policy, in place of the learner's own.

    Before training, the checkpoint's policy plays num_steps steps in every environment from the
    true initial states, and every cell it stands in is counted. An archive slot is then as
    relevant as the share of those steps spent in its cell; cells the policy never visited weigh
    nothing. The checkpoint must come from the same recipe, so its parameters fit the learner's
    network, and the same cell function.
    """

    checkpoint: str
    num_steps: int = 1_000
    temperature: float = 1.0
    seed: int = 0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        archive = unwrap(environment).archive
        init_key, rollout_key = jax.random.split(jax.random.key(self.seed))
        algorithm_state, environment_state, carry, timestep = restore_policy(
            algorithm, environment, self.checkpoint, init_key
        )

        @jax.jit
        def rollout(algorithm_state, environment_state, carry, timestep, key):
            def step(carry_, key):
                environment_state, carry, timestep = carry_
                algorithm_key, environment_key = jax.random.split(key)
                cells = fold(archive.cell_fn(archive_auto_reset_state(environment_state).env_state))
                _, carry, action, _ = algorithm.step(
                    algorithm_state, carry, algorithm_key, timestep, self.temperature
                )
                environment_state, timestep = environment.step(
                    environment_key, environment_state, action
                )
                return (environment_state, carry, timestep), cells

            _, cells = jax.lax.scan(
                step,
                (environment_state, carry, timestep),
                jax.random.split(key, self.num_steps),
            )
            return cells

        cells = jax.device_get(
            rollout(algorithm_state, environment_state, carry, timestep, rollout_key)
        )
        keys, counts = np.unique(np.ravel(cells), return_counts=True)
        self.keys = jnp.asarray(keys, jnp.uint32)
        self.weights = jnp.asarray(counts / counts.sum(), jnp.float32)

    def __call__(
        self,
        successor: SuccessorRepresentation,
        table: Successor,
        reach: Array,
        keys: Array = None,
        mask: Array = None,
    ) -> Array:
        assert hasattr(self, "keys"), "TargetRelevance needs prepare before it is used"
        assert keys is not None, "TargetRelevance needs the archive's keys"
        folded = fold(keys)
        index = jnp.clip(jnp.searchsorted(self.keys, folded), 0, len(self.keys) - 1)
        found = self.keys[index] == folded
        if mask is not None:
            found = found & mask
        return jnp.where(found, self.weights[index], 0.0)


def tree_search(wrapper, algorithm: Algorithm, num_simulations: int, gamma: float, num_considered: int):
    """Gumbel MuZero search over the true simulator, with the current network as prior and leaf value.

    The model is the environment inside the archive wrapper, so rewards are on the scale the critic
    learns. Every simulated step first refreshes the state with the archive's refresh function (a
    new key for PacMan's ghosts), so the tree averages over the simulator's randomness rather than
    replaying one draw. Returns the chosen actions, the search's root values and the critic's values.
    """
    import mctx

    inner = wrapper._env

    def evaluate(params, env_states):
        obs = jax.vmap(inner.observe)(env_states)
        dist, value = algorithm.network.apply(params, obs, temperature=1.0)
        return dist.logits, value[..., 0]

    def recurrent_fn(params, key, action, env_states):
        seed_key, step_key = jax.random.split(key)
        env_states = wrapper.refresh_fn(env_states, seed_key)
        keys = jax.random.split(step_key, jnp.shape(action)[0])
        env_states, timestep = jax.vmap(inner.step)(keys, env_states, action)
        logits, value = evaluate(params, env_states)
        output = mctx.RecurrentFnOutput(
            reward=timestep.reward.astype(jnp.float32),
            discount=gamma * (1.0 - timestep.terminated.astype(jnp.float32)),
            prior_logits=logits,
            value=value,
        )
        return output, env_states

    def search(params, key, env_states):
        logits, value = evaluate(params, env_states)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=env_states)
        output = mctx.gumbel_muzero_policy(
            params,
            key,
            root,
            recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=num_considered,
        )
        return output.action, output.search_tree.summary().value, value

    return search


@struct.dataclass
class SearchGainState(GainState):
    values: Array
    searched_at: Array
    clock: Array
    simulator_steps: Array


@dataclass
class SearchGain(Advantage):
    """How much a short tree search improves on current play: max(0, V_search(s) - V(s)).

    Each update the `num_cells` archive cells searched longest ago (new cells first) are searched
    from their snapshots, `num_simulations` simulator steps each, and their gains replaced. The
    search sees several steps of the true dynamics, so it can find what the critic misses. The
    advantage statistics are still kept, for the noise scale the saturation uses.
    """

    num_cells: int = 256
    num_simulations: int = 32
    num_considered: int = 16

    def init(self, archive_size: int, num_actions: int) -> SearchGainState:
        state = super().init(archive_size, num_actions)
        return SearchGainState(
            **vars(state),
            values=jnp.zeros(archive_size, jnp.float32),
            searched_at=jnp.full(archive_size, -1, jnp.int32),
            clock=jnp.zeros((), jnp.int32),
            simulator_steps=jnp.zeros((), jnp.float32),
        )

    def forget(self, state: SearchGainState, wiped_mask: Array) -> SearchGainState:
        searched_at = jnp.where(wiped_mask, -1, state.searched_at)
        return super().forget(state, wiped_mask).replace(searched_at=searched_at)

    def __call__(self, state: SearchGainState) -> Array:
        return state.values

    def search(self, state: SearchGainState, key: Key, wrapper, archive_state, algorithm, algorithm_state):
        search = tree_search(
            wrapper, algorithm, self.num_simulations, self.gamma, min(self.num_considered, wrapper.num_actions)
        )
        eligible_mask = wrapper.archive.eligible(archive_state)
        staleness = jnp.where(eligible_mask, state.clock - state.searched_at, -1).astype(jnp.float32)
        staleness = staleness + jax.random.uniform(key, staleness.shape) * 0.5  # ties broken at random
        _, slots = jax.lax.top_k(staleness, self.num_cells)
        valid_mask = eligible_mask[slots]
        _, searched, value = search(algorithm_state.params, key, wrapper.archive.take(archive_state, slots))
        improvement = searched - value
        rows = jnp.where(valid_mask, slots, jnp.shape(state.values)[0])
        num_valid = jnp.sum(valid_mask, dtype=jnp.float32)
        lox.log(
            {
                "search/improvement": jnp.sum(jnp.where(valid_mask, improvement, 0.0)) / jnp.maximum(num_valid, 1.0),
                "search/improved": jnp.sum(valid_mask & (improvement > 0.0), dtype=jnp.float32)
                / jnp.maximum(num_valid, 1.0),
                "search/never_searched": jnp.sum(eligible_mask & (state.searched_at < 0), dtype=jnp.float32),
            }
        )
        return state.replace(
            values=state.values.at[rows].set(jnp.maximum(improvement, 0.0), mode="drop"),
            searched_at=state.searched_at.at[rows].set(state.clock, mode="drop"),
            clock=state.clock + 1,
            simulator_steps=state.simulator_steps + num_valid * self.num_simulations,
        )


def returns_to_go(transitions: Transition, gamma: float, return_lambda: float) -> tuple[Array, Array]:
    """The lambda-return from every step of a rollout (lambda 1: rewards to the rollout's end,
    bootstrapped with the critic there), and the critic's values at those steps."""
    values = transitions.aux["value"]
    advantages, _ = generalized_advantage_estimator(
        jax.tree.map(lambda leaf: leaf[:-1], transitions),
        values[:-1],
        values[-1],
        gamma=gamma,
        gae_lambda=return_lambda,
    )
    return advantages + values[:-1], values[:-1]


def upper_quantile(sums: Array, squares: Array, counts: Array, quantile: float) -> Array:
    """A high quantile of the returns seen from each cell, from their decayed sums, assuming they
    are roughly normal."""
    from jax.scipy.special import ndtri

    mean = sums / jnp.maximum(counts, 1e-12)
    spread = jnp.sqrt(jnp.maximum(squares / jnp.maximum(counts, 1e-12) - mean**2, 0.0))
    return mean + ndtri(quantile) * spread


@struct.dataclass
class HindsightState(GainState):
    return_sums: Array
    return_squares: Array
    return_counts: Array
    value_sums: Array
    value_counts: Array


@dataclass
class HindsightGain(Advantage):
    """The return left on the table from a cell: max(0, G_hi(c) - V(c)).

    G_hi is a high `quantile` of the return-to-go observed from the cell, remembered with the slow
    `memory_rate`, so it keeps what earlier and luckier episodes achieved; V is the critic's value
    there, averaged with the gain's `rate`, so it stays current. This is Self-Imitation Learning's
    (R - V)+ used to choose restarts. Under the current policy alone E[G] = V, so the score is the
    critic's lag plus z_quantile times the spread of the returns: the memory is what separates
    "already done better here" from "the returns here are noisy".
    """

    quantile: float = 0.75
    memory_rate: float = 0.02
    return_lambda: float = 1.0

    def init(self, archive_size: int, num_actions: int) -> HindsightState:
        state = super().init(archive_size, num_actions)
        column = jnp.zeros(archive_size, jnp.float32)
        return HindsightState(
            **vars(state),
            return_sums=column,
            return_squares=column,
            return_counts=column,
            value_sums=column,
            value_counts=column,
        )

    def update(self, state: HindsightState, transitions: Transition) -> HindsightState:
        state = super().update(state, transitions)
        returns, values = returns_to_go(transitions, self.gamma, self.return_lambda)
        archive_size, *_ = jnp.shape(state.return_counts)
        cell_slots = transitions.first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~transitions.second.truncated[:-1].astype(bool)
        rows = jnp.reshape(jnp.where(valid_mask, cell_slots, archive_size), (-1,))

        def tally(value):
            return piled(archive_size, rows, jnp.reshape(jnp.where(valid_mask, value, 0.0), (-1,)))

        counts = tally(1.0)
        keep, fresh = 1.0 - self.memory_rate, 1.0 - self.rate
        return state.replace(
            return_sums=keep * state.return_sums + tally(returns),
            return_squares=keep * state.return_squares + tally(returns**2),
            return_counts=keep * state.return_counts + counts,
            value_sums=fresh * state.value_sums + tally(values),
            value_counts=fresh * state.value_counts + counts,
        )

    def __call__(self, state: HindsightState) -> Array:
        best = upper_quantile(state.return_sums, state.return_squares, state.return_counts, self.quantile)
        value = state.value_sums / jnp.maximum(state.value_counts, 1e-12)
        seen = (state.return_counts > 0.0) & (state.value_counts > 0.0)
        return jnp.where(seen, jnp.maximum(best - value, 0.0), 0.0)


class ExpectileHeads(nn.Module):
    """Q(s, a) and two value heads, each on its own copy of the learner's encoder: V_hi, an upper
    expectile of Q over the actions taken, and V_mean, their mean."""

    encoder: nn.Module
    num_actions: int

    def setup(self):
        self.q_encoder = self.encoder.clone()
        self.v_encoder = self.encoder.clone()
        self.q_head = nn.Dense(self.num_actions)
        self.v_head = nn.Dense(2)

    def __call__(self, obs):
        return self.q_head(self.q_encoder(obs)), self.v_head(self.v_encoder(obs))


@struct.dataclass
class ExpectileState(GainState):
    params: PyTree
    target_params: PyTree
    optimizer_state: PyTree
    key: Key
    gap_sums: Array
    gap_counts: Array


@dataclass
class ExpectileGain(Advantage):
    """The expectile gap: replay states where the best things tried beat what is done on average.

    Two extra heads learn from the rollouts PPO already collects, as in implicit Q-learning
    (Kostrikov et al., 2021). Q(s, a) regresses on r + gamma V_hi(s'), a mean backup that averages
    out the environment's luck; V_hi(s) is the `expectile` (0.9) of the target Q over the actions
    actually taken, the value of the best behaviour tried from s. The priority of a cell is the
    average over its visited states of V_hi - V, where V is the 0.5 expectile of the same Q
    (`baseline: mean`, so Q's errors cancel) or PPO's critic (`baseline: critic`).

    The expectile is taken over actions only, not over returns, so a cell whose returns are noisy
    but whose actions all lead to the same mean scores nothing.
    """

    expectile: float = 0.9
    baseline: str = "mean"
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    num_minibatches: int = 8
    polyak: float = 0.005
    seed: int = 0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        assert self.baseline in ("mean", "critic"), self.baseline
        extractor = algorithm.network.feature_extractor
        encoder = getattr(extractor, "observation_extractor", extractor)
        wrapper = unwrap(environment)
        self.heads = ExpectileHeads(encoder=encoder, num_actions=wrapper.num_actions)
        _, timestep = wrapper._env.init(jax.random.key(0))
        self.example = jax.tree.map(lambda leaf: leaf[None], timestep.obs)
        self.optimizer = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(self.lr))

    def init(self, archive_size: int, num_actions: int) -> ExpectileState:
        assert hasattr(self, "heads"), "ExpectileGain needs prepare before it is used"
        state = super().init(archive_size, num_actions)
        init_key, key = jax.random.split(jax.random.key(self.seed))
        params = self.heads.init(init_key, self.example)
        column = jnp.zeros(archive_size, jnp.float32)
        return ExpectileState(
            **vars(state),
            params=params,
            target_params=params,
            optimizer_state=self.optimizer.init(params),
            key=key,
            gap_sums=column,
            gap_counts=column,
        )

    def forget(self, state: ExpectileState, wiped_mask: Array) -> ExpectileState:
        kept = dict(
            params=state.params,
            target_params=state.target_params,
            optimizer_state=state.optimizer_state,
            key=state.key,
        )
        cleared = state.replace(**{name: None for name in kept})
        return super().forget(cleared, wiped_mask).replace(**kept)

    def losses(self, params, target_params, batch):
        obs, next_obs, action, reward, terminated, valid = batch
        q_all, v = self.heads.apply(params, obs)
        _, v_next = self.heads.apply(params, next_obs)
        q = jnp.take_along_axis(q_all, action[:, None], axis=-1)[:, 0]
        backup = jax.lax.stop_gradient(reward + self.gamma * (1.0 - terminated) * v_next[:, 0])
        target_q_all, _ = self.heads.apply(target_params, obs)
        target_q = jax.lax.stop_gradient(jnp.take_along_axis(target_q_all, action[:, None], axis=-1)[:, 0])
        above = target_q - v[:, 0]
        weight = jnp.abs(self.expectile - (above < 0.0).astype(jnp.float32))
        count = jnp.maximum(jnp.sum(valid), 1.0)

        def mean(value):
            return jnp.sum(jnp.where(valid, value, 0.0)) / count

        q_loss = mean((q - backup) ** 2)
        hi_loss = mean(weight * above**2)
        mean_loss = mean((target_q - v[:, 1]) ** 2)
        return q_loss + hi_loss + mean_loss, (q_loss, hi_loss, mean_loss)

    def update(self, state: ExpectileState, transitions: Transition) -> ExpectileState:
        state = super().update(state, transitions)
        first, second = transitions.first, transitions.second

        def flat(leaf):
            leaf = leaf[:-1]
            return jnp.reshape(leaf, (-1, *leaf.shape[2:]))

        obs = jax.tree.map(flat, first.obs)
        batch = (
            obs,
            jax.tree.map(flat, second.obs),
            flat(second.action).astype(jnp.int32),
            flat(second.reward).astype(jnp.float32),
            flat(second.terminated).astype(jnp.float32),
            ~flat(second.truncated).astype(bool),
        )
        size = jnp.shape(batch[2])[0]
        key, permute_key = jax.random.split(state.key)
        order = jax.random.permutation(permute_key, size)[: size // self.num_minibatches * self.num_minibatches]
        minibatches = jax.tree.map(
            lambda leaf: jnp.reshape(leaf[order], (self.num_minibatches, -1, *leaf.shape[1:])), batch
        )

        def train(carry, minibatch):
            params, target_params, optimizer_state = carry
            (_, parts), grads = jax.value_and_grad(self.losses, has_aux=True)(params, target_params, minibatch)
            updates, optimizer_state = self.optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            target_params = optax.incremental_update(params, target_params, self.polyak)
            return (params, target_params, optimizer_state), parts

        (params, target_params, optimizer_state), (q_loss, hi_loss, mean_loss) = jax.lax.scan(
            train, (state.params, state.target_params, state.optimizer_state), minibatches
        )
        _, v = self.heads.apply(params, obs)
        baseline = v[:, 1] if self.baseline == "mean" else flat(transitions.aux["value"])
        gap = v[:, 0] - baseline
        archive_size, *_ = jnp.shape(state.gap_counts)
        cells = flat(first.info["cell"])
        valid = (cells >= 0) & batch[-1]
        rows = jnp.where(valid, cells, archive_size)
        decay = 1.0 - self.rate
        lox.log(
            {
                "expectile/q_loss": jnp.mean(q_loss),
                "expectile/hi_loss": jnp.mean(hi_loss),
                "expectile/mean_loss": jnp.mean(mean_loss),
                "expectile/gap": jnp.sum(jnp.where(valid, gap, 0.0)) / jnp.maximum(jnp.sum(valid), 1),
                "expectile/positive": jnp.sum(valid & (gap > 0.0)) / jnp.maximum(jnp.sum(valid), 1),
            }
        )
        return state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            key=key,
            gap_sums=decay * state.gap_sums + piled(archive_size, rows, jnp.where(valid, gap, 0.0)),
            gap_counts=decay * state.gap_counts + piled(archive_size, rows, valid.astype(jnp.float32)),
        )

    def __call__(self, state: ExpectileState) -> Array:
        gap = state.gap_sums / jnp.maximum(state.gap_counts, 1e-12)
        return jnp.where(state.gap_counts > 0.0, jnp.maximum(gap, 0.0), 0.0)


@struct.dataclass
class ScreenState(TargetState):
    group_counts: Array
    group_sums: Array
    group_targets: Array
    group_returns: Array
    group_values: Array
    clock: Array


@dataclass
class FairScreen(TargetAdvantage):
    """Steers with the plain advantage gain and measures how well cheap gains track the target gain.

    Environments are split by index into six groups that tally advantages, and the target's action
    probabilities, apart. Each estimator is computed twice from disjoint groups, so no candidate
    shares noise with the reference and every pair of copies gives that estimator's split-half
    reliability:

    - target gain, the reference: groups 0 and 1;
    - single greedy, from one group like one half of the double: groups 2 and 4;
    - double greedy: picks in 2 and scores in 3 (and back), and picks in 4 and scores in 5;
    - search gain, every `search_every` updates on `num_cells` well-visited cells: two searches
      with independent keys;
    - hindsight gain, max(0, G_hi - V), from one group like single greedy: groups 2 and 4. Its
      returns are remembered with `memory_rate` as in HindsightGain.

    Logged per visit threshold: the raw correlation of each candidate with the target (averaged
    over the four pairings of copies), the reliabilities, and the correlation corrected for
    attenuation, r_xy / sqrt(r_xx r_yy). The group tallies decay by `screen_rate` per update.
    """

    shrinkage: float = 4.0
    screen_rate: float = 0.05
    thresholds: tuple = (10.0, 50.0)
    num_cells: int = 256
    num_simulations: int = 32
    num_considered: int = 16
    search_every: int = 10
    quantile: float = 0.75
    memory_rate: float = 0.02
    return_lambda: float = 1.0

    def __call__(self, state: ScreenState) -> Array:
        return self.power(state)

    def init(self, archive_size: int, num_actions: int) -> ScreenState:
        state = Advantage.init(self, archive_size, num_actions)
        groups = jnp.zeros((6, archive_size, num_actions), jnp.float32)
        return ScreenState(
            **vars(state),
            target_sums=jnp.zeros((archive_size, num_actions), jnp.float32),
            target=self.target,
            group_counts=groups,
            group_sums=groups,
            group_targets=groups,
            # Per group and cell: decayed count, sum and sum of squares of the returns, and the
            # count and sum of the critic's values.
            group_returns=jnp.zeros((6, archive_size, 3), jnp.float32),
            group_values=jnp.zeros((6, archive_size, 2), jnp.float32),
            clock=jnp.zeros((), jnp.int32),
        )

    def update(self, state: ScreenState, transitions: Transition) -> ScreenState:
        state = super().update(state, transitions)
        first, second = transitions.first, transitions.second
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimator(
            jax.tree.map(lambda leaf: leaf[:-1], transitions),
            values[:-1],
            values[-1],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        _, archive_size, num_actions = jnp.shape(state.group_counts)
        cell_slots = first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~second.truncated[:-1].astype(bool)
        actions = jnp.clip(second.action[:-1].astype(jnp.int32), 0, num_actions - 1)
        group = jnp.broadcast_to(jnp.arange(cell_slots.shape[-1]) % 6, cell_slots.shape)
        rows = jnp.where(valid_mask, group * archive_size + cell_slots, 6 * archive_size)
        size = 6 * archive_size * num_actions
        bins = jnp.reshape(jnp.where(valid_mask, rows * num_actions + actions, size), (-1,))

        def tally(value):
            summed = piled(size, bins, jnp.reshape(jnp.where(valid_mask, value, 0.0), (-1,)))
            return jnp.reshape(summed, (6, archive_size, num_actions))

        _, _, _, aux = self.algorithm.step(
            state.target, transitions.carry, jax.random.key(self.seed), first, 1.0
        )
        target_probs = jax.nn.softmax(aux["logits"][:-1].astype(jnp.float32), axis=-1)
        targets = (
            jnp.zeros((6 * archive_size, num_actions))
            .at[jnp.reshape(rows, (-1,))]
            .add(jnp.reshape(target_probs, (-1, num_actions)), mode="drop")
            .reshape(6, archive_size, num_actions)
        )
        returns, visited_values = returns_to_go(transitions, self.gamma, self.return_lambda)
        flat_rows = jnp.reshape(rows, (-1,))

        def by_cell(value):
            summed = piled(6 * archive_size, flat_rows, jnp.reshape(jnp.where(valid_mask, value, 0.0), (-1,)))
            return jnp.reshape(summed, (6, archive_size))

        visits = by_cell(1.0)
        decay = 1.0 - self.screen_rate
        keep = 1.0 - self.memory_rate
        return state.replace(
            group_counts=decay * state.group_counts + tally(1.0),
            group_sums=decay * state.group_sums + tally(advantages),
            group_targets=decay * state.group_targets + targets,
            group_returns=keep * state.group_returns
            + jnp.stack([visits, by_cell(returns), by_cell(returns**2)], axis=-1),
            group_values=decay * state.group_values
            + jnp.stack([visits, by_cell(visited_values)], axis=-1),
        )

    def estimates(self, state: ScreenState) -> dict:
        counts, sums = state.group_counts, state.group_sums
        present = counts > 0.0
        means = jnp.where(present, sums / jnp.maximum(counts, 1e-12), 0.0)
        visits = jnp.sum(counts, axis=-1)
        centred = jnp.where(present, means - (jnp.sum(sums, -1) / jnp.maximum(visits, 1e-12))[..., None], 0.0)
        shrunk = counts / (counts + self.shrinkage) * centred
        probs = state.group_targets / jnp.maximum(visits, 1e-12)[..., None]
        target = jnp.maximum(jnp.sum(jnp.where(present, probs * centred, 0.0), axis=-1), 0.0)
        single = jnp.max(jnp.where(present, shrunk, 0.0), axis=-1)
        picked = jnp.argmax(jnp.where(present, shrunk, -jnp.inf), axis=-1)

        def scored(pick, score):
            value = jnp.take_along_axis(shrunk[score], picked[pick][..., None], axis=-1)[..., 0]
            return jnp.where(jnp.any(present[pick], axis=-1), value, 0.0)

        def double(a, b):
            return jnp.maximum(0.5 * (scored(a, b) + scored(b, a)), 0.0)

        counted, summed, squared = jnp.moveaxis(state.group_returns, -1, 0)
        best = upper_quantile(summed, squared, counted, self.quantile)
        value_counts, value_sums = jnp.moveaxis(state.group_values, -1, 0)
        value = value_sums / jnp.maximum(value_counts, 1e-12)
        hindsight = jnp.where(
            (counted > 0.0) & (value_counts > 0.0), jnp.maximum(best - value, 0.0), 0.0
        )
        return {
            "target": (target[0], target[1]),
            "single": (single[2], single[4]),
            "double": (double(2, 3), double(4, 5)),
            "hindsight": (hindsight[2], hindsight[4]),
        }

    def report(self, copies: dict, mask: Array, suffix: str) -> dict:
        reference = copies["target"]
        reliability = {name: rank_correlation(*pair, mask) for name, pair in copies.items()}
        metrics = {f"screen/reliability_target{suffix}": reliability["target"]}
        for name, (a, b) in copies.items():
            if name == "target":
                continue
            raw = 0.25 * sum(rank_correlation(x, y, mask) for x in (a, b) for y in reference)
            floor = jnp.sqrt(jnp.maximum(reliability[name] * reliability["target"], 0.0))
            metrics[f"screen/{name}_target{suffix}"] = raw
            metrics[f"screen/reliability_{name}{suffix}"] = reliability[name]
            metrics[f"screen/corrected_{name}{suffix}"] = jnp.where(floor > 0.05, raw / jnp.maximum(floor, 1e-6), jnp.nan)
        metrics[f"screen/cells{suffix}"] = jnp.sum(mask, dtype=jnp.float32)
        return metrics

    def screen(self, state: ScreenState, archive_mask: Array) -> None:
        copies = self.estimates(state)
        group_visits = jnp.sum(state.group_counts, axis=-1)
        metrics = {}
        for threshold in self.thresholds:
            mask = archive_mask & (jnp.sum(group_visits, axis=0) >= threshold)
            metrics.update(self.report(copies, mask, f"_{threshold:g}"))
        lox.log(metrics)

    def search(self, state: ScreenState, key: Key, wrapper, archive_state, algorithm, algorithm_state):
        def measure():
            search = tree_search(
                wrapper, algorithm, self.num_simulations, self.gamma, min(self.num_considered, wrapper.num_actions)
            )
            pick_key, first_key, second_key = jax.random.split(key, 3)
            visits = jnp.sum(state.group_counts, axis=(0, -1))
            well_mask = wrapper.archive.eligible(archive_state) & (visits >= min(self.thresholds))
            draw = jnp.where(well_mask, jax.random.uniform(pick_key, visits.shape), -1.0)
            _, slots = jax.lax.top_k(draw, self.num_cells)
            snapshots = wrapper.archive.take(archive_state, slots)

            def gain(search_key):
                _, searched, value = search(algorithm_state.params, search_key, snapshots)
                return jnp.maximum(searched - value, 0.0)

            copies = {name: (a[slots], b[slots]) for name, (a, b) in self.estimates(state).items()}
            copies["search"] = (gain(first_key), gain(second_key))
            return self.report(copies, well_mask[slots], "_searched")

        due = state.clock % self.search_every == 0
        shapes = jax.eval_shape(measure)
        metrics = jax.lax.cond(
            due, measure, lambda: jax.tree.map(lambda leaf: jnp.full(leaf.shape, jnp.nan, leaf.dtype), shapes)
        )
        # Updates without a search log nothing, so the epoch's average holds only measured ones.
        lox.log(jax.tree.map(lambda value: jnp.where(due, value, jnp.nan), metrics))
        return state.replace(clock=state.clock + 1)


def forecast_logits(now_logits: Array, old_logits: Array, kappa: float, clip: float) -> Array:
    """log pi_fore = log pi_t + kappa (log pi_t - log pi_old), renormalised; the step is clipped so
    actions near zero probability cannot blow it up."""
    now = jax.nn.log_softmax(now_logits.astype(jnp.float32), axis=-1)
    old = jax.nn.log_softmax(old_logits.astype(jnp.float32), axis=-1)
    return jax.nn.log_softmax(now + kappa * jnp.clip(now - old, -clip, clip), axis=-1)


@struct.dataclass
class ForecastState(GainState):
    snapshots: PyTree
    cursor: Array
    filled: Array
    fore_sums: Array
    now_sums: Array


@dataclass
class ForecastAdvantage(Advantage):
    """The oracle's gain with the reference policy replaced by the policy's own extrapolation.

    pi_fore(a|s) is proportional to pi_t(a|s) (pi_t(a|s) / pi_{t-k}(a|s))^kappa, pi_{t-k} the
    parameters `horizon` updates ago, kept in a ring of snapshots. By the performance-difference
    lemma a cell gains sum_a (pi_fore - pi_t)(a|s) A(s, a): both policies' probabilities are averaged
    over the states the learner visits in the cell, and A is the cell's per-action advantage. When
    the policy stops changing pi_fore = pi_t and every cell scores zero.
    """

    horizon: int = 10
    kappa: float = 1.0
    clip: float = 5.0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        self.algorithm = algorithm
        _, timestep = unwrap(environment)._env.init(jax.random.key(0))
        timestep = jax.tree.map(lambda leaf: leaf[None], timestep)
        algorithm_state, _ = algorithm.init(jax.random.key(0), timestep)
        self.example = algorithm_state.params

    def init(self, archive_size: int, num_actions: int) -> ForecastState:
        assert hasattr(self, "example"), "ForecastAdvantage needs prepare before it is used"
        state = super().init(archive_size, num_actions)
        table = jnp.zeros((archive_size, num_actions), jnp.float32)
        return ForecastState(
            **vars(state),
            snapshots=jax.tree.map(lambda leaf: jnp.stack([leaf] * self.horizon), self.example),
            cursor=jnp.zeros((), jnp.int32),
            filled=jnp.zeros((), jnp.int32),
            fore_sums=table,
            now_sums=table,
        )

    def forget(self, state: ForecastState, wiped_mask: Array) -> ForecastState:
        snapshots = state.snapshots
        return super().forget(state.replace(snapshots=None), wiped_mask).replace(snapshots=snapshots)

    def observe_params(self, state: ForecastState, algorithm_state) -> ForecastState:
        params = algorithm_state.params
        # The first snapshot fills the whole ring, so the forecast starts at zero instead of
        # extrapolating from the random initial network.
        snapshots = jax.tree.map(
            lambda ring, leaf: jnp.where(
                state.filled == 0, jnp.broadcast_to(leaf, ring.shape), ring.at[state.cursor].set(leaf)
            ),
            state.snapshots,
            params,
        )
        return state.replace(
            snapshots=snapshots,
            cursor=(state.cursor + 1) % self.horizon,
            filled=state.filled + 1,
        )

    def old_params(self, state: ForecastState) -> PyTree:
        # The ring's cursor points at the oldest snapshot, the one written horizon updates ago.
        return jax.tree.map(lambda ring: ring[state.cursor], state.snapshots)

    def update(self, state: ForecastState, transitions: Transition) -> ForecastState:
        state = super().update(state, transitions)
        first = transitions.first
        archive_size, num_actions = jnp.shape(state.action_counts)
        now_logits = transitions.aux["logits"][:-1]
        dist, _ = self.algorithm.network.apply(
            self.old_params(state), jax.tree.map(lambda leaf: leaf[:-1], first.obs), temperature=1.0
        )
        fore = jnp.exp(forecast_logits(now_logits, dist.logits, self.kappa, self.clip))
        now = jax.nn.softmax(now_logits.astype(jnp.float32), axis=-1)
        cell_slots = first.info["cell"][:-1]
        valid_mask = (cell_slots >= 0) & ~transitions.second.truncated[:-1].astype(bool)
        rows = jnp.reshape(jnp.where(valid_mask, cell_slots, archive_size), (-1,))

        def tally(probs):
            return jnp.zeros((archive_size, num_actions)).at[rows].add(
                jnp.reshape(probs, (-1, num_actions)), mode="drop"
            )

        moved = jnp.sum(jnp.abs(fore - now), axis=-1)
        lox.log(
            {
                "forecast/moved": jnp.sum(jnp.where(valid_mask, moved, 0.0))
                / jnp.maximum(jnp.sum(valid_mask), 1),
            }
        )
        decay = 1.0 - self.rate
        return state.replace(
            fore_sums=decay * state.fore_sums + tally(fore),
            now_sums=decay * state.now_sums + tally(now),
        )

    def __call__(self, state: ForecastState) -> Array:
        present_mask = state.action_counts > 0.0
        visits = jnp.maximum(state.cell_visits, 1e-12)[:, None]
        change = (state.fore_sums - state.now_sums) / visits
        gain = jnp.sum(jnp.where(present_mask, change * self.centre(state), 0.0), axis=-1)
        return jnp.maximum(gain, 0.0)


@struct.dataclass
class PlayedState:
    env_states: PyTree
    earned: Array
    visits: Array
    key: Key
    simulator_steps: Array


@dataclass
class SearchRelevance:
    """Where a few from-start episodes played with search go, in place of the learner's visitation.

    `num_envs` side environments, outside the training batch, play from the true start with the
    search's chosen action, `num_steps` real steps per update, and start over when an episode ends.
    The archive cells they stand in are counted, decayed by `rate` per update; a slot is as relevant
    as its share. States outside the archive count toward the miss rate only. Until the first visit
    is counted the learner's own visitation stands in.
    """

    num_envs: int = 64
    num_steps: int = 8
    num_simulations: int = 32
    num_considered: int = 16
    gamma: float = 0.99
    rate: float = 0.01
    seed: int = 0

    def prepare(self, algorithm: Algorithm, environment: Environment) -> None:
        self.wrapper = unwrap(environment)

    def init(self, archive_size: int) -> PlayedState:
        init_key, key = jax.random.split(jax.random.key(self.seed))
        env_states, _ = jax.vmap(self.wrapper._env.init)(jax.random.split(init_key, self.num_envs))
        return PlayedState(
            env_states=env_states,
            earned=jnp.zeros(self.num_envs, jnp.float32),
            visits=jnp.zeros(archive_size, jnp.float32),
            key=key,
            simulator_steps=jnp.zeros((), jnp.float32),
        )

    def play(self, state: PlayedState, wrapper, archive_state, algorithm, algorithm_state, **context) -> PlayedState:
        from boonta.utils.table import find

        inner = wrapper._env
        search = tree_search(
            wrapper, algorithm, self.num_simulations, self.gamma, min(self.num_considered, wrapper.num_actions)
        )
        archive_size, *_ = jnp.shape(state.visits)

        def step(carry, key):
            env_states, earned = carry
            search_key, seed_key, step_key, reset_key = jax.random.split(key, 4)
            slots = find(
                archive_state.keys,
                archive_state.mask,
                wrapper.archive.cell_fn(env_states),
                wrapper.archive.num_probes,
            )
            action, _, _ = search(algorithm_state.params, search_key, env_states)
            env_states = wrapper.refresh_fn(env_states, seed_key)
            env_states, timestep = jax.vmap(inner.step)(
                jax.random.split(step_key, self.num_envs), env_states, action
            )
            earned = earned + timestep.reward
            done = timestep.terminated | timestep.truncated
            fresh, _ = jax.vmap(inner.init)(jax.random.split(reset_key, self.num_envs))
            env_states = jax.tree.map(
                lambda new, live: jnp.where(jnp.reshape(done, done.shape + (1,) * (live.ndim - 1)), new, live),
                fresh,
                env_states,
            )
            lox.log({"search/episode_return": jnp.where(done, earned, jnp.nan)})
            return (env_states, jnp.where(done, 0.0, earned)), slots

        key, scan_key = jax.random.split(state.key)
        (env_states, earned), slots = jax.lax.scan(
            step, (state.env_states, state.earned), jax.random.split(scan_key, self.num_steps)
        )
        counts = jnp.zeros(archive_size).at[jnp.where(slots >= 0, slots, archive_size)].add(1.0, mode="drop")
        lox.log({"search/relevance_miss_rate": jnp.mean((slots < 0).astype(jnp.float32))})
        return state.replace(
            env_states=env_states,
            earned=earned,
            visits=(1.0 - self.rate) * state.visits + counts,
            key=key,
            simulator_steps=state.simulator_steps
            + self.num_envs * self.num_steps * (self.num_simulations + 1),
        )

    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, reach: Array, played: PlayedState = None, **cells
    ) -> Array:
        visits = played.visits
        total = jnp.sum(visits)
        return jnp.where(
            total > 0.0, visits / jnp.maximum(total, 1e-12), reach / jnp.maximum(jnp.sum(reach), 1e-12)
        )


@dataclass
class ForecastRelevance(SearchRelevance):
    """Where the forecast policy goes from the start: `num_envs` probe environments, outside the
    training batch, play pi_fore for `num_steps` steps per update, and the archive cells they stand
    in are counted, decayed by `rate`. Needs the ForecastAdvantage gain, which keeps the snapshots."""

    num_steps: int = 16
    kappa: float = 1.0
    clip: float = 5.0

    def play(self, state: PlayedState, wrapper, archive_state, algorithm, algorithm_state, gain_state=None, **context):
        from boonta.utils.table import find

        assert isinstance(gain_state, ForecastState), "ForecastRelevance needs the forecast gain"
        inner = wrapper._env
        old_params = jax.tree.map(lambda ring: ring[gain_state.cursor], gain_state.snapshots)
        archive_size, *_ = jnp.shape(state.visits)

        def step(carry, key):
            env_states, earned = carry
            act_key, seed_key, step_key, reset_key = jax.random.split(key, 4)
            slots = find(
                archive_state.keys,
                archive_state.mask,
                wrapper.archive.cell_fn(env_states),
                wrapper.archive.num_probes,
            )
            obs = jax.vmap(inner.observe)(env_states)
            now, _ = algorithm.network.apply(algorithm_state.params, obs, temperature=1.0)
            old, _ = algorithm.network.apply(old_params, obs, temperature=1.0)
            logits = forecast_logits(now.logits, old.logits, self.kappa, self.clip)
            action = jax.random.categorical(act_key, logits)
            env_states = wrapper.refresh_fn(env_states, seed_key)
            env_states, timestep = jax.vmap(inner.step)(
                jax.random.split(step_key, self.num_envs), env_states, action
            )
            earned = earned + timestep.reward
            done = timestep.terminated | timestep.truncated
            fresh, _ = jax.vmap(inner.init)(jax.random.split(reset_key, self.num_envs))
            env_states = jax.tree.map(
                lambda new, live: jnp.where(jnp.reshape(done, done.shape + (1,) * (live.ndim - 1)), new, live),
                fresh,
                env_states,
            )
            lox.log({"forecast/episode_return": jnp.where(done, earned, jnp.nan)})
            return (env_states, jnp.where(done, 0.0, earned)), slots

        key, scan_key = jax.random.split(state.key)
        (env_states, earned), slots = jax.lax.scan(
            step, (state.env_states, state.earned), jax.random.split(scan_key, self.num_steps)
        )
        counts = jnp.zeros(archive_size).at[jnp.where(slots >= 0, slots, archive_size)].add(1.0, mode="drop")
        lox.log({"forecast/relevance_miss_rate": jnp.mean((slots < 0).astype(jnp.float32))})
        return state.replace(
            env_states=env_states,
            earned=earned,
            visits=(1.0 - self.rate) * state.visits + counts,
            key=key,
            simulator_steps=state.simulator_steps + self.num_envs * self.num_steps,
        )


class Diagonal:
    def init(self, algorithm_state: PyTree) -> PyTree:
        return None

    def direct(self, selector, algorithm, algorithm_state, transitions) -> PyTree:
        return None

    def __call__(self, selector, state, wrapper, archive_auto_reset_state, **context) -> Array:
        return selector.relevant(state) * selector.gain(state.gain)


@dataclass
class Kernel:
    preconditioned: bool = True

    def init(self, algorithm_state: PyTree) -> PyTree:
        return jax.tree.map(jnp.zeros_like, algorithm_state.params["params"])

    def policy(self, algorithm: Algorithm, params: PyTree):
        def policy(inner, obs):
            dist, _ = algorithm.network.apply({**params, "params": inner}, obs, temperature=1.0)
            return dist

        return policy

    def direct(
        self, selector, algorithm: Algorithm, algorithm_state: PyTree, transitions: Transition
    ) -> PyTree:
        params = algorithm_state.params
        policy = self.policy(algorithm, params)
        direction = jax.grad(lambda inner: self.climb(selector, policy, inner, transitions))(
            params["params"]
        )
        if self.preconditioned:
            direction = precondition(direction, algorithm_state.optimizer_state)
        return direction

    def __call__(
        self,
        selector,
        state,
        wrapper,
        archive_auto_reset_state,
        algorithm: Algorithm | None = None,
        algorithm_state=None,
        **context,
    ) -> Array:
        params = algorithm_state.params
        inner = params["params"]
        policy = self.policy(algorithm, params)
        obs = wrapper.observe_cells(archive_auto_reset_state.archive_state)
        present_mask = state.gain.action_counts > 0.0
        advantages = jnp.where(present_mask, selector.gain.centre(state.gain), 0.0)
        weight = jax.lax.stop_gradient(jnp.exp(policy(inner, obs).logits) * advantages)
        _, aligned = jax.jvp(
            lambda inner: jnp.sum(weight * policy(inner, obs).logits, axis=-1),
            (inner,),
            (state.direction,),
        )
        return jnp.where(archive_auto_reset_state.archive_state.mask, aligned, 0.0)

    def climb(self, selector, policy, inner, transitions: Transition) -> Array:
        first, second = transitions.first, transitions.second
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimator(
            jax.tree.map(lambda leaf: leaf[:-1], transitions),
            values[:-1],
            values[-1],
            gamma=selector.gain.gamma,
            gae_lambda=selector.gain.gae_lambda,
        )
        rho_mask = first.info["rho"][:-1] & ~second.truncated[:-1].astype(bool)
        log_probs = policy(inner, jax.tree.map(lambda leaf: leaf[:-1], first.obs)).log_prob(
            second.action[:-1]
        )
        total = jnp.sum(jnp.where(rho_mask, jax.lax.stop_gradient(advantages) * log_probs, 0.0))
        return total / jnp.maximum(jnp.sum(rho_mask, dtype=jnp.float32), 1.0)


@dataclass
class Leverage:
    """Restarts chosen to maximise the log-determinant of the batch's feature covariance.

    phi is the network's penultimate layer (the feature extractor's output, d features) and
    Sigma = E_batch[phi phi^T] its uncentred second moment over the last rollout, measured with the
    parameters just updated. A state's leverage l(s) = phi(s)^T Sigma^-1 phi(s) averages exactly d
    over the batch, and d log det Sigma / d(weight on s) = l(s) - d, so restarting where l / d is
    largest is a Frank-Wolfe step towards the D-optimal (by Kiefer-Wolfowitz also G-optimal) design,
    the one that minimises the worst-case relative condition number max_s l(s) of Q-NPG's analysis.
    A cell is worth l / d of its snapshot; the footprint then sums it along the segment a restart
    there would run. No value or relevance enters. `ridge` is relative to the mean eigenvalue.
    """

    dimension: int = 256
    ridge: float = 1e-3
    batch_size: int = 2048

    def features(self, algorithm: Algorithm, params: PyTree, obs) -> Array:
        def penultimate(module, obs):
            hidden = module.feature_extractor(obs)
            assert module.torso is None, "Leverage reads the feature extractor of networks without a torso"
            return hidden

        return algorithm.network.apply(params, obs, method=penultimate).astype(jnp.float32)

    def init(self, algorithm_state: PyTree) -> PyTree:
        # Until the first rollout is measured, every feature counts as equally covered.
        return jnp.eye(self.dimension, dtype=jnp.float32)

    def direct(self, selector, algorithm: Algorithm, algorithm_state: PyTree, transitions: Transition) -> PyTree:
        obs = jax.tree.map(lambda leaf: leaf[:-1], transitions.first.obs)
        phi = self.features(algorithm, algorithm_state.params, obs)
        phi = jnp.reshape(phi, (-1, phi.shape[-1]))
        valid = jnp.reshape(~transitions.second.truncated[:-1].astype(bool), (-1,))
        count = jnp.maximum(jnp.sum(valid, dtype=jnp.float32), 1.0)
        weighted = jnp.where(valid[:, None], phi, 0.0)
        second = weighted.T @ weighted / count
        dimension = second.shape[0]
        second = second + self.ridge * jnp.trace(second) / dimension * jnp.eye(dimension)
        eigenvalues = jnp.clip(jnp.linalg.eigvalsh(second), 1e-30)
        # The rank of the centred covariance: ReLU features share a large mean, which would
        # otherwise fill the first direction of the uncentred second moment.
        mean = jnp.sum(weighted, axis=0) / count
        centred = jnp.clip(jnp.linalg.eigvalsh(weighted.T @ weighted / count - jnp.outer(mean, mean)), 1e-30)

        def effective_rank(values):
            share = values / jnp.sum(values)
            return jnp.exp(-jnp.sum(share * jnp.log(share)))

        lox.log(
            {
                "leverage/effective_rank": effective_rank(centred),
                "leverage/uncentred_rank": effective_rank(eigenvalues),
                "leverage/log_det": jnp.sum(jnp.log(eigenvalues)),
                "leverage/dead_features": jnp.mean((jnp.diag(second) < 1e-6 * jnp.trace(second) / dimension).astype(jnp.float32)),
            }
        )
        return jnp.linalg.cholesky(second)

    def __call__(
        self,
        selector,
        state,
        wrapper,
        archive_auto_reset_state,
        algorithm: Algorithm | None = None,
        algorithm_state=None,
        **context,
    ) -> Array:
        archive_state = archive_auto_reset_state.archive_state
        mask = archive_state.mask
        chol = state.direction
        dimension = chol.shape[0]
        snapshots = archive_state.cell_states.snapshot
        inner = wrapper._env

        def leverage_of(snapshot):
            phi = self.features(algorithm, algorithm_state.params, jax.tree.map(lambda leaf: leaf[None], inner.observe(snapshot)))[0]
            solved = jax.scipy.linalg.solve_triangular(chol, phi, lower=True)
            return jnp.sum(solved**2)

        leverage = jax.lax.map(leverage_of, snapshots, batch_size=self.batch_size)
        ratio = leverage / dimension
        count = jnp.maximum(jnp.sum(mask, dtype=jnp.float32), 1.0)
        lox.log(
            {
                "leverage/archive_mean": jnp.sum(jnp.where(mask, ratio, 0.0)) / count,
                "leverage/archive_max": jnp.max(jnp.where(mask, ratio, 0.0)),
                "leverage/under_covered": jnp.sum(mask & (ratio > 1.0), dtype=jnp.float32) / count,
            }
        )
        return jnp.where(mask, ratio, 0.0)


def precondition(direction: PyTree, optimizer_state: PyTree) -> PyTree:
    adams = [
        leaf
        for leaf in jax.tree.leaves(
            optimizer_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState)
        )
        if isinstance(leaf, optax.ScaleByAdamState)
    ]
    if not adams:
        return direction
    adam, *_ = adams
    correction = 1.0 - 0.999 ** jnp.maximum(adam.count, 1)
    return jax.tree.map(
        lambda gradient, second: gradient / (jnp.sqrt(second / correction) + 1e-8),
        direction,
        adam.nu,
    )


def moments(value: Array, over_mask: Array) -> tuple[Array, Array]:
    count = jnp.maximum(jnp.sum(over_mask, dtype=value.dtype), 1.0)
    centre = jnp.sum(jnp.where(over_mask, value, 0.0)) / count
    return centre, jnp.sqrt(jnp.sum(jnp.where(over_mask, (value - centre) ** 2, 0.0)) / count)


def random_argmax(value: Array, mask: Array, key: Key, shape) -> Array:
    highest = jnp.max(jnp.where(mask, value, -jnp.inf))
    return pick(key, jnp.where(mask & (value >= highest), 0.0, -jnp.inf), shape)


class Marginal:
    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        def choose(visits, key):
            factor = selector.diminishing_factor(state, visits)
            marginal = selector.footprint.pour(selector.successor, state.successor, worth * factor)
            chosen_slot = random_argmax(marginal, eligible_mask, key, ())
            spread = selector.footprint.expected_visits(
                selector.successor, state.successor, chosen_slot
            )
            return visits + spread, chosen_slot

        _, selected_slots = jax.lax.scan(
            choose,
            selector.committed_visits(state, resident_counts),
            jax.random.split(key, count),
        )
        return selected_slots


class Uniform:
    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (count,))


@dataclass
class WaterFilling:
    iterations: int = 50
    noise_scale_multiplier: float = 1.0

    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        argmax_key, order_key = jax.random.split(key)
        noise_scale = state.gain.scale * self.noise_scale_multiplier
        committed = selector.committed_visits(state, resident_counts)
        first_value = jnp.where(eligible_mask, selector.value(state, worth, committed), 0.0)
        curvature = selector.footprint.pour_squared(
            selector.successor,
            state.successor,
            jnp.abs(worth) / (1.0 + committed / jnp.maximum(noise_scale, 1e-30)) ** 3,
        )
        saturating_restarts = noise_scale * first_value / jnp.maximum(curvature, 1e-30)
        highest = jnp.max(first_value)

        def restarts_at(level):
            return jnp.where(
                first_value > level,
                saturating_restarts * (jnp.sqrt(first_value / level) - 1.0),
                0.0,
            )

        def bisect(_, bounds):
            low, high = bounds
            middle = jnp.sqrt(low * high)
            too_many = jnp.sum(restarts_at(middle)) > count
            return jnp.where(too_many, middle, low), jnp.where(too_many, high, middle)

        low, _ = jax.lax.fori_loop(
            0, self.iterations, bisect, (highest * 1e-12, highest)
        )
        restarts = restarts_at(low)
        boundaries = jnp.cumsum(restarts) * count / jnp.maximum(jnp.sum(restarts), 1e-30)
        archive_size, *_ = jnp.shape(restarts)
        filled_slots = jnp.minimum(
            jnp.searchsorted(boundaries, jnp.arange(count) + 0.5), archive_size - 1
        )
        filled_slots = jax.random.permutation(order_key, filled_slots)
        return jnp.where(
            (noise_scale > 0.0) & (highest > 0.0),
            filled_slots,
            random_argmax(first_value, eligible_mask, argmax_key, (count,)),
        )


@dataclass
class Importance:
    gain_exponent: float = 0.5

    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        relevance = selector.relevant(state)
        gain = jnp.maximum(selector.gain(state.gain), 0.0)
        weight = relevance * gain**self.gain_exponent
        logits = jnp.where(eligible_mask & (weight > 0.0), jnp.log(jnp.maximum(weight, 1e-30)), -jnp.inf)
        fallback = jnp.where(eligible_mask, 0.0, -jnp.inf)
        logits = jnp.where(jnp.any(jnp.isfinite(logits)), logits, fallback)
        return jax.random.categorical(key, logits, shape=(count,))


@dataclass
class Blend:
    inner: "Marginal | WaterFilling | TopK | Softmax"
    uniform_share: float = 0.5

    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        uniform_key, inner_key = jax.random.split(key)
        num_uniform = int(round(self.uniform_share * count))
        uniform_slots = Uniform()(selector, state, worth, eligible_mask, resident_counts, uniform_key, num_uniform)
        inner_slots = self.inner(
            selector, state, worth, eligible_mask, resident_counts, inner_key, count - num_uniform
        )
        return jnp.concatenate([uniform_slots, inner_slots])


@dataclass
class TopK:
    width: int = 32

    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        poured = selector.footprint.pour(selector.successor, state.successor, worth)
        archive_size, *_ = jnp.shape(poured)
        top_values, top_slots = jax.lax.top_k(
            jnp.where(eligible_mask, poured, -jnp.inf), min(self.width, archive_size)
        )
        num_valid = jnp.maximum(jnp.sum(jnp.isfinite(top_values)), 1)
        return top_slots[jnp.arange(count) % num_valid]


@dataclass
class Softmax:
    temperature: float = 0.1

    def __call__(self, selector, state, worth, eligible_mask, resident_counts, key, count):
        poured = selector.footprint.pour(selector.successor, state.successor, worth)
        highest = jnp.max(jnp.where(eligible_mask, poured, -jnp.inf))
        logits = poured / (jnp.maximum(jnp.abs(highest), 1e-30) * self.temperature)
        return jax.random.categorical(
            key, jnp.where(eligible_mask, logits, -jnp.inf), shape=(count,)
        )


@dataclass
class Selector:
    successor: SuccessorRepresentation
    footprint: SuccessorFootprint | UniformFootprint
    relevance: CountedRelevance | UniformRelevance | TargetRelevance
    gain: Advantage | GreedyAdvantage | UniformGain | ModelAdvantage | TargetAdvantage | SearchGain
    score: Diagonal | Kernel | Leverage = Diagonal()
    allocation: Marginal | WaterFilling | Uniform | TopK | Softmax | Blend | Importance = Marginal()
    k: float = 4.0
    rate: float = 0.2
    gate: bool = False

    def __post_init__(self):
        assert self.k >= 1.0, f"k must be at least 1, got {self.k}"

    def num_placed(self, num_envs: int) -> int:
        return int(round((1.0 - 1.0 / self.k) * num_envs))

    def init(self, archive_size: int, num_actions: int) -> SelectorState:
        return SelectorState(
            successor=self.successor.init(archive_size),
            reach=jnp.zeros(archive_size),
            gain=self.gain.init(archive_size, num_actions),
            played=self.relevance.init(archive_size) if hasattr(self.relevance, "play") else None,
        )

    def search(self, state: SelectorState, key: Key, wrapper, archive_auto_reset_state, algorithm, algorithm_state):
        """Runs the tree searches of a search gain or relevance, and logs the simulator steps they cost."""
        gain_key, play_key = jax.random.split(key)
        archive_state = archive_auto_reset_state.archive_state
        spent = jnp.zeros((), jnp.float32)
        if hasattr(self.gain, "search"):
            state = state.replace(
                gain=self.gain.search(state.gain, gain_key, wrapper, archive_state, algorithm, algorithm_state)
            )
            spent = spent + getattr(state.gain, "simulator_steps", 0.0)
        if hasattr(self.relevance, "play"):
            state = state.replace(
                played=self.relevance.play(
                    state.played, wrapper, archive_state, algorithm, algorithm_state, gain_state=state.gain
                )
            )
            spent = spent + state.played.simulator_steps
        if hasattr(self.gain, "search") or hasattr(self.relevance, "play"):
            lox.log({"search/simulator_steps": spent})
        return state

    def update(self, state: SelectorState, key: Key, transitions: Transition) -> SelectorState:
        first, second = transitions.first, transitions.second
        leaving_slots, landing_slots = first.info["cell"], second.info["cell"]
        steps, envs = jnp.shape(leaving_slots)
        done = jnp.reshape(second.done, (steps, envs, -1)).all(axis=-1)
        archive_size, *_ = jnp.shape(state.reach)

        def tread(carry, step):
            table, visit_counts = carry
            leaving_slots, landing_slots, done, evicted_slots, rho_mask, key = step
            visit_rows = jnp.where(rho_mask & (leaving_slots >= 0), leaving_slots, archive_size)
            visit_counts = visit_counts.at[visit_rows].add(1.0, mode="drop")
            wiped_mask = wipe(evicted_slots, archive_size)
            table = self.successor.forget(table, wiped_mask)
            table = self.successor.backup(
                table, leaving_slots, landing_slots, ~done & (landing_slots >= 0), key
            )
            return (table, jnp.where(wiped_mask, 0.0, visit_counts)), wiped_mask

        (table, visit_counts), wiped_mask = jax.lax.scan(
            tread,
            (state.successor, jnp.zeros_like(state.reach)),
            (
                leaving_slots,
                landing_slots,
                done,
                second.info["evicted"],
                first.info["rho"],
                jax.random.split(key, steps),
            ),
        )
        wiped_mask = jnp.any(wiped_mask, axis=0)
        reach = jnp.where(wiped_mask, 0.0, (1.0 - self.rate) * state.reach)
        reach = reach + self.rate * visit_counts
        return state.replace(
            successor=table,
            reach=reach,
            gain=self.gain.update(self.gain.forget(state.gain, wiped_mask), transitions),
        )

    def select(
        self, state: SelectorState, wrapper, archive_auto_reset_state, key: Key, **context
    ):
        archive_state = archive_auto_reset_state.archive_state
        state = state.replace(keys=archive_state.keys, mask=archive_state.mask)
        eligible_mask = wrapper.archive.eligible(archive_state)
        num_placed = self.num_placed(wrapper.num_envs)
        resident_counts = wrapper.count_residents(archive_auto_reset_state, num_placed)
        worth = self.score(self, state, wrapper, archive_auto_reset_state, **context)
        selected_slots = self.allocation(
            self, state, worth, eligible_mask, resident_counts, key, num_placed
        )
        self.watch(state, worth, archive_state.mask, selected_slots)
        self.saturation(state, worth, eligible_mask, resident_counts, selected_slots)
        self.log_relative_efficiency(
            state, worth, eligible_mask, selected_slots, wrapper.num_envs
        )
        if self.gate:
            value_per_step, start_value_per_step = self.per_step(state, worth)
            admitted_mask = value_per_step[selected_slots] > start_value_per_step
            lox.log({"mu/admitted": jnp.mean(admitted_mask.astype(jnp.float32))})
            selected_slots = jnp.where(admitted_mask, selected_slots, -1)
        return selected_slots

    def per_step(self, state: SelectorState, worth: Array) -> tuple[Array, Array]:
        archive_size, *_ = jnp.shape(state.reach)
        episode_lengths = self.footprint.pour(
            self.successor, state.successor, jnp.ones(archive_size)
        )
        value_per_step = self.value(state, worth) / jnp.maximum(episode_lengths, 1e-30)
        start_value_per_step = jnp.sum(state.reach * worth) / jnp.maximum(
            jnp.sum(state.reach), 1e-30
        )
        return value_per_step, start_value_per_step

    def window_weights(self, state: SelectorState, wrapper, archive_auto_reset_state, selected_slots) -> Array:
        num_placed, *_ = jnp.shape(selected_slots)
        num_envs = wrapper.num_envs
        archive_size, *_ = jnp.shape(state.reach)
        relevance = state.reach / jnp.maximum(jnp.sum(state.reach), 1e-30)
        placed = jnp.sum(selected_slots >= 0)
        restart_share = jnp.zeros(archive_size).at[
            jnp.where(selected_slots >= 0, selected_slots, archive_size)
        ].add(1.0, mode="drop") / jnp.maximum(placed, 1)
        placed_fraction = placed / num_envs
        mixture = (1.0 - placed_fraction) * relevance + placed_fraction * restart_share
        block_mask = jnp.arange(num_envs) >= num_envs - num_placed
        start_slots = jnp.where(
            block_mask,
            jnp.zeros(num_envs, jnp.int32).at[num_envs - num_placed:].set(selected_slots),
            archive_auto_reset_state.current_slots,
        )
        known = start_slots >= 0
        clipped = jnp.clip(start_slots, 0)
        weights = relevance[clipped] / jnp.maximum(mixture[clipped], 1e-30)
        return jnp.where(known, weights, 1.0).astype(jnp.float32)

    def relevant(self, state: SelectorState) -> Array:
        played = {} if state.played is None else {"played": state.played}
        return self.relevance(
            self.successor, state.successor, state.reach, keys=state.keys, mask=state.mask, **played
        )

    def diminishing_factor(self, state: SelectorState, visits: Array | float) -> Array:
        scale = state.gain.scale
        return jnp.where(scale > 0.0, 1.0 / (1.0 + visits / jnp.maximum(scale, 1e-30)) ** 2, 1.0)

    def committed_visits(self, state: SelectorState, resident_counts: Array | float) -> Array:
        return self.footprint.spill(
            self.successor, state.successor, jnp.zeros_like(state.reach) + resident_counts
        )

    def value(self, state: SelectorState, worth: Array, visits: Array | float = 0.0) -> Array:
        term = worth * self.diminishing_factor(state, visits)
        return self.footprint.pour(self.successor, state.successor, term)

    def saturation(
        self,
        state: SelectorState,
        worth: Array,
        eligible_mask: Array,
        resident_counts: Array | float,
        selected_slots: Array,
    ) -> None:
        archive_size, *_ = jnp.shape(eligible_mask)
        committed = self.committed_visits(state, resident_counts)
        restart_counts = jnp.zeros(archive_size).at[selected_slots].add(1.0)
        visits = committed + self.footprint.spill(self.successor, state.successor, restart_counts)
        scale = jnp.maximum(state.gain.scale, 1e-30)
        before = jnp.where(eligible_mask, self.value(state, worth, committed), -jnp.inf)
        best, runner_up = jax.lax.top_k(before, 2)[0]
        chosen_slot = jnp.argmax(restart_counts)
        after = self.value(state, worth, visits)[chosen_slot]
        peak_visits = jnp.max(visits)
        lox.log(
            {
                "mu/peak_load": peak_visits / scale,
                "mu/peak_factor": self.diminishing_factor(state, peak_visits),
                "archive/top_ratio": best / jnp.maximum(runner_up, 1e-30),
                "mu/chosen_discount": after / jnp.maximum(before[chosen_slot], 1e-30),
            }
        )

    def log_relative_efficiency(
        self,
        state: SelectorState,
        worth: Array,
        eligible_mask: Array,
        selected_slots: Array,
        num_envs: int,
    ) -> None:
        archive_size, *_ = jnp.shape(eligible_mask)
        num_placed, *_ = jnp.shape(selected_slots)
        num_starting = max(num_envs - num_placed, 1)
        start_visits = state.reach
        start_steps = jnp.sum(start_visits)
        restarts_per_slot = jnp.zeros(archive_size).at[selected_slots].add(1.0)
        restart_footprint = self.footprint.spill(self.successor, state.successor, restarts_per_slot)
        restart_visits = (
            restart_footprint
            * start_steps
            * num_placed
            / num_starting
            / jnp.maximum(jnp.sum(restart_footprint), 1e-30)
        )
        mixed_visits = start_visits + restart_visits
        rho_only_visits = start_visits * num_envs / num_starting
        noise_scale = state.gain.scale

        def improvement_from(visits, saturated):
            if saturated:
                visits = jnp.where(
                    noise_scale > 0.0, noise_scale * visits / (visits + noise_scale), visits
                )
            return jnp.sum(worth * visits)

        def ratio(numerator, denominator):
            return jnp.where(denominator > 0.0, numerator / denominator, 1.0)

        value_per_step, start_value_per_step = self.per_step(state, worth)
        best_value_per_step = jnp.max(jnp.where(eligible_mask, value_per_step, 0.0))
        numerator = improvement_from(mixed_visits, saturated=False)
        denominator = improvement_from(rho_only_visits, saturated=False)
        lox.log(
            {
                "relative_efficiency/linear": ratio(numerator, denominator),
                "relative_efficiency/saturated": ratio(
                    improvement_from(mixed_visits, saturated=True),
                    improvement_from(rho_only_visits, saturated=True),
                ),
                "relative_efficiency/ceiling": ratio(
                    best_value_per_step, start_value_per_step
                ),
                "relative_efficiency/numerator": numerator,
                "relative_efficiency/denominator": denominator,
            }
        )

    def watch(
        self, state: SelectorState, worth: Array, archive_mask: Array, selected_slots: Array
    ) -> None:
        archive_size, *_ = jnp.shape(archive_mask)
        num_cells = jnp.maximum(jnp.sum(archive_mask, dtype=jnp.float32), 1.0)
        weights = jnp.zeros(archive_size).at[selected_slots].add(
            1.0 / max(jnp.shape(selected_slots)[0], 1)
        )
        entropy = -jnp.sum(jnp.where(weights > 0.0, weights * jnp.log(weights), 0.0))
        value = self.value(state, worth)
        gain = self.gain(state.gain)
        relevance = self.relevant(state)
        screen = getattr(self.gain, "screen", None)
        if screen is not None:
            screen(state.gain, archive_mask)
        lox.log(
            {
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
                "mu/rho": jnp.sum(jnp.where(state.reach > 0.0, weights, 0.0)),
            }
        )


@dataclass
class PSR(Pit):
    environment: ArchiveAutoReset
    selector: Selector
    algorithm: Algorithm
    restart_at: str = "rollout"
    importance_weights: bool = False

    def __post_init__(self):
        assert isinstance(
            self.environment, ArchiveAutoReset
        ), "PSR restarts from an ArchiveAutoReset, and the environment wraps none"
        assert self.restart_at in ("rollout", "episode"), self.restart_at

    def init(self, key: Key, state: PyTree) -> SelectorState:
        ledger = self.selector.init(
            self.environment.archive.archive_size, self.environment.num_actions
        )
        return ledger.replace(direction=self.selector.score.init(state.algorithm_state))

    def update(
        self,
        ledger: SelectorState,
        key: Key,
        transitions: Transition,
        algorithm_state: PyTree,
    ) -> SelectorState:
        ledger = self.selector.update(ledger, key, transitions)
        if hasattr(self.selector.gain, "observe_params"):
            ledger = ledger.replace(gain=self.selector.gain.observe_params(ledger.gain, algorithm_state))
        direction = self.selector.score.direct(
            self.selector, self.algorithm, algorithm_state, transitions
        )
        return ledger.replace(direction=direction)

    def __call__(self, key: Key, state):
        select_key, assign_key, search_key = jax.random.split(key, 3)
        ledger = self.selector.search(
            state.pit_state,
            search_key,
            self.environment,
            archive_auto_reset_state(state.environment_state),
            self.algorithm,
            state.algorithm_state,
        )
        state = state.replace(pit_state=ledger)

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                slots = self.selector.select(
                    state.pit_state,
                    self.environment,
                    node,
                    select_key,
                    algorithm=self.algorithm,
                    algorithm_state=state.algorithm_state,
                )
                weights = None
                if self.importance_weights:
                    weights = self.selector.window_weights(state.pit_state, self.environment, node, slots)
                return self.environment.update(
                    node, assign_key, slots=slots, preempt=self.restart_at == "rollout", weights=weights
                )
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm,
    environment: Environment,
    selector: Selector,
    restart_at: str = "rollout",
    importance_weights: bool = False,
    **kwargs,
) -> tuple[Algorithm, Environment, PSR, Lap | None]:
    for part in (selector.relevance, selector.gain):
        if hasattr(part, "prepare"):
            part.prepare(algorithm, environment)
    pit = PSR(unwrap(environment), selector, algorithm, restart_at, importance_weights)
    return algorithm, environment, pit, None
