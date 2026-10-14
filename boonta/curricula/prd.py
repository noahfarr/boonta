from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimation
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, share, unwrap)
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
class GreedyAdvantage(Advantage):
    shrinkage: float = 4.0

    def __call__(self, state: GainState) -> Array:
        present_mask = state.action_counts > 0.0
        trust = state.action_counts / (state.action_counts + self.shrinkage)
        return jnp.max(jnp.where(present_mask, trust * self.centre(state), 0.0), axis=-1)


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


class CountedRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, reach: Array
    ) -> Array:
        return reach / jnp.maximum(jnp.sum(reach), 1e-12)


class UniformRelevance:
    def __call__(
        self, successor: SuccessorRepresentation, table: Successor, reach: Array
    ) -> Array:
        return jnp.ones_like(reach)


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
        advantages, _ = generalized_advantage_estimation(
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
    relevance: CountedRelevance | UniformRelevance
    gain: Advantage | GreedyAdvantage | ModelAdvantage
    score: Diagonal | Kernel = Diagonal()
    allocation: Marginal | TopK | Softmax = Marginal()
    k: float = 4.0
    rate: float = 0.2

    def init(self, archive_size: int, num_actions: int) -> SelectorState:
        return SelectorState(
            successor=self.successor.init(archive_size),
            reach=jnp.zeros(archive_size),
            gain=self.gain.init(archive_size, num_actions),
        )

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
        eligible_mask = wrapper.archive.eligible(archive_state)
        num_placed = share(self.k, wrapper.num_envs)
        resident_counts = wrapper.count_residents(archive_auto_reset_state, num_placed)
        worth = self.score(self, state, wrapper, archive_auto_reset_state, **context)
        selected_slots = self.allocation(
            self, state, worth, eligible_mask, resident_counts, key, num_placed
        )
        self.watch(state, worth, archive_state.mask, selected_slots)
        self.saturation(state, worth, eligible_mask, resident_counts, selected_slots)
        return selected_slots

    def relevant(self, state: SelectorState) -> Array:
        return self.relevance(self.successor, state.successor, state.reach)

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
class PRD(Pit):
    environment: ArchiveAutoReset
    selector: Selector
    algorithm: Algorithm

    def __post_init__(self):
        assert isinstance(
            self.environment, ArchiveAutoReset
        ), "PRD restarts from an ArchiveAutoReset, and the environment wraps none"

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
        direction = self.selector.score.direct(
            self.selector, self.algorithm, algorithm_state, transitions
        )
        return ledger.replace(direction=direction)

    def __call__(self, key: Key, state):
        select_key, assign_key = jax.random.split(key)

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
                return self.environment.update(node, assign_key, slots=slots)
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm, environment: Environment, selector: Selector, **kwargs
) -> tuple[Algorithm, Environment, PRD, Lap | None]:
    return algorithm, environment, PRD(unwrap(environment), selector, algorithm), None
