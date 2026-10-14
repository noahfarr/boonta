from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimation
from boonta.environments.wrappers import UED
from boonta.environments.wrappers.ued import UEDState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Transition
from boonta.utils.typing import Environment


@struct.dataclass
class LevelStats:
    episodes: Array
    regret: Array
    value: Array
    returns: Array


def positive_value_loss(stats: LevelStats, returns: Array) -> Array:
    return stats.regret / jnp.maximum(stats.episodes, 1.0)


def maximum_monte_carlo(stats: LevelStats, returns: Array) -> Array:
    best = jnp.maximum(returns, stats.returns)
    return best - stats.value / jnp.maximum(stats.episodes, 1.0)


@struct.dataclass
class LevelBuffer:
    levels: PyTree
    scores: Array
    timestamps: Array
    returns: Array
    num_levels: Array
    episodes: Array
    exploring: Array
    assignment: Array


@dataclass
class PLR(Pit):
    environment: UED
    buffer_size: int
    num_envs: int
    replay_probability: float
    staleness_coefficient: float
    temperature: float
    min_length: int
    gamma: float
    gae_lambda: float
    generate_level: Callable[[Key], PyTree]
    score: Callable[[LevelStats, Array], Array]

    @property
    def num_slots(self) -> int:
        return self.buffer_size + self.num_envs

    def score_episodes(
        self, transitions: Transition, values: Array, advantages: Array
    ) -> LevelStats:
        def accumulate_episode(carry, step):
            positive, value, reward, length = carry
            advantage, current, earned, done = step
            positive = positive + jnp.maximum(advantage, 0.0)
            value = value + current
            reward = reward + earned
            length = length + 1.0
            finished = (positive / length, value / length, reward)
            kept = 1.0 - done
            return (positive * kept, value * kept, reward * kept, length * kept), finished

        zeros = jnp.zeros(values.shape[1:], jnp.float32)
        done = transitions.second.done.astype(jnp.float32)
        _, (regret, value, returns) = jax.lax.scan(
            accumulate_episode,
            (zeros, zeros, zeros, zeros),
            (
                advantages.astype(jnp.float32),
                values.astype(jnp.float32),
                transitions.second.reward.astype(jnp.float32),
                done,
            ),
        )
        theta = transitions.second.info["theta"]
        index = jnp.where((done > 0) & (theta >= 0), theta, self.num_slots).reshape(-1)
        empty = jnp.zeros(self.num_slots, jnp.float32)
        return LevelStats(
            episodes=empty.at[index].add(1.0, mode="drop"),
            regret=empty.at[index].add(regret.reshape(-1), mode="drop"),
            value=empty.at[index].add(value.reshape(-1), mode="drop"),
            returns=jnp.full(self.num_slots, -jnp.inf, jnp.float32)
            .at[index]
            .max(returns.reshape(-1), mode="drop"),
        )

    def rank_weights(self, buffer: LevelBuffer) -> Array:
        buffer_size, *_ = buffer.scores.shape
        valid = jnp.arange(buffer_size) < buffer.num_levels
        order = jnp.argsort(-jnp.where(valid, buffer.scores, -jnp.inf))
        ranks = jnp.empty_like(order).at[order].set(jnp.arange(buffer_size) + 1)
        weights = jnp.where(valid, 1.0 / ranks, 0.0) ** (1.0 / self.temperature)
        total = weights.sum()
        return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), 0.0)

    def staleness_weights(self, buffer: LevelBuffer) -> Array:
        buffer_size, *_ = buffer.timestamps.shape
        valid = jnp.arange(buffer_size) < buffer.num_levels
        staleness = jnp.where(valid, buffer.episodes - buffer.timestamps, 0)
        staleness = staleness.astype(jnp.float32)
        total = staleness.sum()
        return jnp.where(
            total > 0,
            staleness / jnp.where(total > 0, total, 1.0),
            valid / jnp.maximum(buffer.num_levels, 1),
        )

    def replay_weights(self, buffer: LevelBuffer) -> Array:
        rank = self.rank_weights(buffer)
        staleness = self.staleness_weights(buffer)
        coefficient = self.staleness_coefficient
        return (1.0 - coefficient) * rank + coefficient * staleness

    def update_scores(
        self, buffer: LevelBuffer, stats: LevelStats, scored: Array
    ) -> LevelBuffer:
        buffer_size, *_ = buffer.scores.shape
        finished = (stats.episodes[:buffer_size] > 0) & (
            jnp.arange(buffer_size) < buffer.num_levels
        )
        return buffer.replace(
            scores=jnp.where(finished, scored[:buffer_size], buffer.scores),
            returns=jnp.where(
                finished,
                jnp.maximum(buffer.returns, stats.returns[:buffer_size]),
                buffer.returns,
            ),
        )

    def admit(
        self, buffer: LevelBuffer, stats: LevelStats, scored: Array
    ) -> tuple[LevelBuffer, Array]:
        buffer_size, *_ = buffer.scores.shape
        num_slots, *_ = scored.shape
        exploring = buffer.exploring

        def insert_level(carry, source):
            buffer, num_admitted = carry
            target = jnp.where(
                buffer.num_levels < buffer_size,
                buffer.num_levels,
                jnp.argmin(self.replay_weights(buffer)),
            )
            replaced = exploring & (buffer.scores[target] < scored[source])
            episodes = buffer.episodes + exploring.astype(jnp.int32)
            buffer = buffer.replace(
                scores=buffer.scores.at[target].set(
                    jnp.where(replaced, scored[source], buffer.scores[target])
                ),
                returns=buffer.returns.at[target].set(
                    jnp.where(replaced, stats.returns[source], buffer.returns[target])
                ),
                timestamps=buffer.timestamps.at[target].set(
                    jnp.where(replaced, episodes, buffer.timestamps[target])
                ),
                num_levels=jnp.where(
                    replaced,
                    jnp.minimum(buffer.num_levels + 1, buffer_size),
                    buffer.num_levels,
                ),
                episodes=episodes,
                levels=jax.tree.map(
                    lambda leaf: leaf.at[target].set(
                        jnp.where(replaced, leaf[source], leaf[target])
                    ),
                    buffer.levels,
                ),
            )
            return (buffer, num_admitted + replaced.astype(jnp.int32)), None

        (buffer, num_admitted), _ = jax.lax.scan(
            insert_level,
            (buffer, jnp.int32(0)),
            jnp.arange(buffer_size, num_slots),
        )
        return buffer, num_admitted

    def explore(self, buffer: LevelBuffer, key: Key) -> LevelBuffer:
        fresh = jax.vmap(self.generate_level)(jax.random.split(key, self.num_envs))
        return buffer.replace(
            levels=jax.tree.map(
                lambda leaf, new: leaf.at[self.buffer_size :].set(new),
                buffer.levels,
                fresh,
            ),
            exploring=jnp.bool_(True),
            assignment=self.buffer_size
            + jnp.arange(self.num_envs, dtype=jnp.int32),
        )

    def replay(self, buffer: LevelBuffer, key: Key) -> LevelBuffer:
        buffer_size, *_ = buffer.scores.shape

        def pick_level(buffer, key):
            weights = self.replay_weights(buffer)
            index = jax.random.choice(key, buffer_size, p=weights)
            index = index.astype(jnp.int32)
            episodes = buffer.episodes + 1
            buffer = buffer.replace(
                timestamps=buffer.timestamps.at[index].set(episodes),
                episodes=episodes,
            )
            return buffer, index

        buffer, assignment = jax.lax.scan(
            pick_level, buffer, jax.random.split(key, self.num_envs)
        )
        return buffer.replace(exploring=jnp.bool_(False), assignment=assignment)

    def init(self, key: Key, state: PyTree) -> LevelBuffer:
        buffer = LevelBuffer(
            levels=jax.tree.map(
                lambda leaf: jnp.zeros((self.num_slots, *leaf.shape), leaf.dtype),
                jax.eval_shape(self.generate_level, key),
            ),
            scores=jnp.full(self.buffer_size, -jnp.inf, jnp.float32),
            timestamps=jnp.zeros(self.buffer_size, jnp.int32),
            returns=jnp.full(self.buffer_size, -jnp.inf, jnp.float32),
            num_levels=jnp.int32(0),
            episodes=jnp.int32(0),
            exploring=jnp.bool_(True),
            assignment=jnp.zeros(self.num_envs, jnp.int32),
        )
        return self.explore(buffer, key)

    def update(
        self,
        buffer: LevelBuffer,
        key: Key,
        transitions: Transition,
        algorithm_state: PyTree,
    ) -> LevelBuffer:
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimation(
            transitions, values, values[-1], self.gamma, self.gae_lambda
        )
        stats = self.score_episodes(
            jax.tree.map(lambda leaf: leaf[1:], transitions),
            values[1:],
            advantages[1:],
        )
        returns = jnp.concatenate(
            [buffer.returns, jnp.full(self.num_envs, -jnp.inf, jnp.float32)]
        )
        scored = jnp.where(stats.episodes > 0, self.score(stats, returns), -jnp.inf)
        buffer = self.update_scores(buffer, stats, scored)
        buffer, num_admitted = self.admit(buffer, stats, scored)

        decide_key, replay_key, explore_key = jax.random.split(key, 3)
        replaying = (buffer.num_levels >= self.min_length) & (
            jax.random.uniform(decide_key) < self.replay_probability
        )
        buffer = jax.lax.cond(
            replaying,
            lambda buffer: self.replay(buffer, replay_key),
            lambda buffer: self.explore(buffer, explore_key),
            buffer,
        )
        self.report(buffer, stats, scored, num_admitted)
        return buffer

    def report(
        self, buffer: LevelBuffer, stats: LevelStats, scored: Array, num_admitted
    ) -> None:
        buffer_size = self.buffer_size
        valid = jnp.arange(buffer_size) < buffer.num_levels
        finished = stats.episodes.sum()
        fresh = stats.episodes[buffer_size:] > 0
        lox.log(
            {
                "buffer/num_levels": buffer.num_levels,
                "buffer/num_admitted": num_admitted,
                "buffer/score": jnp.where(valid, buffer.scores, 0.0).sum()
                / jnp.maximum(buffer.num_levels, 1),
                "mu/score": jnp.where(
                    valid, self.replay_weights(buffer) * buffer.scores, 0.0
                ).sum(),
                "mu/fraction": stats.episodes[:buffer_size].sum()
                / jnp.maximum(finished, 1.0),
                "rho/score": jnp.where(
                    fresh.any(),
                    jnp.where(fresh, scored[buffer_size:], 0.0).sum()
                    / jnp.maximum(fresh.sum(), 1),
                    jnp.nan,
                ),
            }
        )

    def __call__(self, key: Key, state):
        buffer = state.pit_state

        def assign(node):
            if isinstance(node, UEDState):
                return self.environment.update(
                    node, key, theta=buffer.levels, assign=buffer.assignment
                )
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, UEDState),
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm,
    environment: Environment,
    buffer_size: int,
    replay_probability: float,
    staleness_coefficient: float,
    temperature: float,
    min_length: int,
    gamma: float,
    gae_lambda: float,
    generate_level: Callable[[Key], PyTree],
    score: Callable[[LevelStats, Array], Array] = positive_value_loss,
    **kwargs,
) -> tuple[Algorithm, UED, PLR, Lap | None]:
    pit = PLR(
        UED(environment),
        buffer_size=buffer_size,
        num_envs=environment.num_envs,
        replay_probability=replay_probability,
        staleness_coefficient=staleness_coefficient,
        temperature=temperature,
        min_length=min_length,
        gamma=gamma,
        gae_lambda=gae_lambda,
        generate_level=generate_level,
        score=score,
    )
    return algorithm, pit.environment, pit, None
