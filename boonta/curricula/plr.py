from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimation
from boonta.environments.wrappers import UED, Wrapper, WrapperState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Timestep, Transition
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


def rank_weights(scores: Array, num_levels: Array, temperature: float) -> Array:
    buffer_size, *_ = scores.shape
    valid = jnp.arange(buffer_size) < num_levels
    order = jnp.argsort(-jnp.where(valid, scores, -jnp.inf))
    ranks = jnp.empty_like(order).at[order].set(jnp.arange(buffer_size) + 1)
    weights = jnp.where(valid, 1.0 / ranks, 0.0) ** (1.0 / temperature)
    total = weights.sum()
    return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), 0.0)


def staleness_weights(timestamps: Array, num_levels: Array, episodes: Array) -> Array:
    buffer_size, *_ = timestamps.shape
    valid = jnp.arange(buffer_size) < num_levels
    staleness = jnp.where(valid, episodes - timestamps, 0).astype(jnp.float32)
    total = staleness.sum()
    return jnp.where(
        total > 0,
        staleness / jnp.where(total > 0, total, 1.0),
        valid / jnp.maximum(num_levels, 1),
    )


@struct.dataclass
class LevelBufferState(WrapperState):
    scores: Array = struct.field(metadata={"axis": None})
    timestamps: Array = struct.field(metadata={"axis": None})
    returns: Array = struct.field(metadata={"axis": None})
    num_levels: Array = struct.field(metadata={"axis": None})
    episodes: Array = struct.field(metadata={"axis": None})
    exploring: Array = struct.field(metadata={"axis": None})


class LevelBuffer(Wrapper):
    def __init__(
        self,
        env,
        buffer_size: int,
        staging_size: int,
        sample: Callable[[Key], PyTree],
    ):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.staging_size = staging_size
        self.sample = sample

    def init(self, key: Key) -> tuple[LevelBufferState, Timestep]:
        buffer_key, env_key = jax.random.split(key)
        env_state, timestep = self._env.init(env_key)
        num_slots = self.buffer_size + self.staging_size
        levels = jax.tree.map(
            lambda leaf: jnp.zeros((num_slots, *leaf.shape), leaf.dtype),
            jax.eval_shape(self.sample, key),
        )
        env_state = self._env.update(
            env_state,
            jax.random.fold_in(buffer_key, 1),
            theta=levels,
            weights=jnp.zeros(num_slots, jnp.float32),
        )
        state = LevelBufferState(
            env_state=env_state,
            scores=jnp.full(self.buffer_size, -jnp.inf, jnp.float32),
            timestamps=jnp.zeros(self.buffer_size, jnp.int32),
            returns=jnp.full(self.buffer_size, -jnp.inf, jnp.float32),
            num_levels=jnp.int32(0),
            episodes=jnp.int32(0),
            exploring=jnp.bool_(False),
        )
        return state, timestep

    def step(
        self, key: Key, state: LevelBufferState, action: Array
    ) -> tuple[LevelBufferState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        return state.replace(env_state=env_state), timestep


@dataclass
class PLR(Pit):
    environment: LevelBuffer
    replay_probability: float
    staleness: float
    temperature: float
    minimum_fill: float
    gamma: float
    gae_lambda: float
    sample: Callable[[Key], PyTree]
    score: Callable[[LevelStats, Array], Array]

    @property
    def num_slots(self) -> int:
        return self.environment.buffer_size + self.environment.staging_size

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

    def replay_weights(self, buffer: LevelBufferState) -> Array:
        rank = rank_weights(buffer.scores, buffer.num_levels, self.temperature)
        staleness = staleness_weights(
            buffer.timestamps, buffer.num_levels, buffer.episodes
        )
        return (1.0 - self.staleness) * rank + self.staleness * staleness

    def update_scores(
        self, buffer: LevelBufferState, stats: LevelStats, scored: Array
    ) -> LevelBufferState:
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

    def draw(
        self, buffer: LevelBufferState, key: Key, count: int
    ) -> tuple[LevelBufferState, Array]:
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

        return jax.lax.scan(pick_level, buffer, jax.random.split(key, count))

    def admit(
        self,
        buffer: LevelBufferState,
        levels,
        stats: LevelStats,
        scored: Array,
        exploring: Array,
    ):
        buffer_size, *_ = buffer.scores.shape
        num_slots, *_ = scored.shape

        def insert_level(carry, source):
            buffer, levels, num_admitted = carry
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
            )
            levels = jax.tree.map(
                lambda leaf: leaf.at[target].set(
                    jnp.where(replaced, leaf[source], leaf[target])
                ),
                levels,
            )
            return (buffer, levels, num_admitted + replaced.astype(jnp.int32)), None

        (buffer, levels, num_admitted), _ = jax.lax.scan(
            insert_level,
            (buffer, levels, jnp.int32(0)),
            jnp.arange(buffer_size, num_slots),
        )
        return buffer, levels, num_admitted

    def init(self, key: Key, state: PyTree) -> LevelStats:
        return LevelStats(
            episodes=jnp.zeros(self.num_slots, jnp.float32),
            regret=jnp.zeros(self.num_slots, jnp.float32),
            value=jnp.zeros(self.num_slots, jnp.float32),
            returns=jnp.full(self.num_slots, -jnp.inf, jnp.float32),
        )

    def update(
        self, state: LevelStats, key: Key, transitions: Transition, algorithm_state: PyTree
    ) -> LevelStats:
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimation(
            transitions, values, values[-1], self.gamma, self.gae_lambda
        )
        return self.score_episodes(
            jax.tree.map(lambda leaf: leaf[1:], transitions),
            values[1:],
            advantages[1:],
        )

    def __call__(self, key: Key, state):
        def curate(node):
            if isinstance(node, LevelBufferState):
                return self.curate(node, state.pit_state, key)
            return node

        environment_state = jax.tree.map(
            curate,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, LevelBufferState),
        )
        return state.replace(environment_state=environment_state)

    def curate(self, buffer: LevelBufferState, stats: LevelStats, key: Key):
        buffer_size = self.environment.buffer_size
        staging_size = self.environment.staging_size
        returns = jnp.concatenate(
            [buffer.returns, jnp.full(staging_size, -jnp.inf, jnp.float32)]
        )
        scored = jnp.where(stats.episodes > 0, self.score(stats, returns), -jnp.inf)

        buffer = self.update_scores(buffer, stats, scored)
        buffer, levels, num_admitted = self.admit(
            buffer, buffer.theta, stats, scored, buffer.exploring
        )

        update_key, decide_key, draw_key, sample_key = jax.random.split(key, 4)
        replaying = (buffer.num_levels / buffer_size >= self.minimum_fill) & (
            jax.random.uniform(decide_key) < self.replay_probability
        )

        def replay(buffer, levels):
            buffer, assignment = self.draw(buffer, draw_key, staging_size)
            return buffer, levels, assignment

        def explore(buffer, levels):
            fresh = jax.vmap(self.sample)(
                jax.random.split(sample_key, staging_size)
            )
            levels = jax.tree.map(
                lambda leaf, new: leaf.at[buffer_size:].set(new), levels, fresh
            )
            assignment = buffer_size + jnp.arange(staging_size, dtype=jnp.int32)
            return buffer, levels, assignment

        buffer, levels, assignment = jax.lax.cond(
            replaying, replay, explore, buffer, levels
        )
        level_weights = self.replay_weights(buffer)

        valid = jnp.arange(buffer_size) < buffer.num_levels
        finished = stats.episodes.sum()
        fresh = stats.episodes[buffer_size:] > 0
        lox.log(
            {
                "archive/num_levels": buffer.num_levels,
                "archive/num_admitted": num_admitted,
                "archive/score": jnp.where(valid, buffer.scores, 0.0).sum()
                / jnp.maximum(buffer.num_levels, 1),
                "mu/score": jnp.where(valid, level_weights * buffer.scores, 0.0).sum(),
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
        return self.environment.update(
            buffer.replace(exploring=~replaying),
            update_key,
            theta=levels,
            assign=assignment,
        )


def plr(
    algorithm: Algorithm,
    environment: Environment,
    buffer_size: int,
    replay_probability: float,
    staleness: float,
    temperature: float,
    minimum_fill: float,
    gamma: float,
    gae_lambda: float,
    sample: Callable[[Key], PyTree],
    score: Callable[[LevelStats, Array], Array] = positive_value_loss,
    **kwargs,
) -> tuple[Algorithm, LevelBuffer, PLR, Lap | None]:
    environment = LevelBuffer(
        UED(environment), buffer_size, environment.num_envs, sample
    )
    pit = PLR(
        environment,
        replay_probability=replay_probability,
        staleness=staleness,
        temperature=temperature,
        minimum_fill=minimum_fill,
        gamma=gamma,
        gae_lambda=gae_lambda,
        sample=sample,
        score=score,
    )
    return algorithm, environment, pit, None
