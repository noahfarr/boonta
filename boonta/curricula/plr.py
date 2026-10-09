from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.curricula.curriculum import within
from boonta.algorithms.advantage_estimators import \
    generalized_advantage_estimation
from boonta.algorithms.wrappers.wrapper import Wrapper as AlgorithmWrapper
from boonta.algorithms.wrappers.wrapper import \
    WrapperState as AlgorithmWrapperState
from boonta.environments.wrappers import UED, Wrapper, WrapperState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Timestep, Transition
from boonta.utils.typing import Environment


@struct.dataclass
class Tally:
    episodes: Array
    regret: Array
    value: Array
    returns: Array


def blank(slots: int) -> Tally:
    zeros = jnp.zeros(slots, jnp.float32)
    return Tally(
        episodes=zeros,
        regret=zeros,
        value=zeros,
        returns=jnp.full(slots, -jnp.inf, jnp.float32),
    )


def tally(
    transitions: Transition, values: Array, advantages: Array, slots: int
) -> Tally:
    def accumulate(carry, step):
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
        accumulate,
        (zeros, zeros, zeros, zeros),
        (
            advantages.astype(jnp.float32),
            values.astype(jnp.float32),
            transitions.second.reward.astype(jnp.float32),
            done,
        ),
    )
    theta = transitions.second.info["theta"]
    index = jnp.where((done > 0) & (theta >= 0), theta, slots).reshape(-1)
    empty = blank(slots)
    return Tally(
        episodes=empty.episodes.at[index].add(1.0, mode="drop"),
        regret=empty.regret.at[index].add(regret.reshape(-1), mode="drop"),
        value=empty.value.at[index].add(value.reshape(-1), mode="drop"),
        returns=empty.returns.at[index].max(returns.reshape(-1), mode="drop"),
    )


def positive_value_loss(tally: Tally, returns: Array) -> Array:
    return tally.regret / jnp.maximum(tally.episodes, 1.0)


def maximum_monte_carlo(tally: Tally, returns: Array) -> Array:
    best = jnp.maximum(returns, tally.returns)
    return best - tally.value / jnp.maximum(tally.episodes, 1.0)


def rank(scores: Array, size: Array, temperature: float) -> Array:
    (capacity,) = scores.shape
    valid = jnp.arange(capacity) < size
    order = jnp.argsort(-jnp.where(valid, scores, -jnp.inf))
    ranks = jnp.empty_like(order).at[order].set(jnp.arange(capacity) + 1)
    weights = jnp.where(valid, 1.0 / ranks, 0.0) ** (1.0 / temperature)
    total = weights.sum()
    return jnp.where(total > 0, weights / jnp.where(total > 0, total, 1.0), 0.0)


def stale(timestamps: Array, size: Array, episodes: Array) -> Array:
    (capacity,) = timestamps.shape
    valid = jnp.arange(capacity) < size
    staleness = jnp.where(valid, episodes - timestamps, 0).astype(jnp.float32)
    total = staleness.sum()
    return jnp.where(
        total > 0,
        staleness / jnp.where(total > 0, total, 1.0),
        valid / jnp.maximum(size, 1),
    )


def weigh(
    scores: Array,
    timestamps: Array,
    size: Array,
    episodes: Array,
    temperature: float,
    staleness: float,
) -> Array:
    return (1.0 - staleness) * rank(scores, size, temperature) + staleness * stale(
        timestamps, size, episodes
    )


@struct.dataclass
class LevelBufferState(WrapperState):
    scores: Array = struct.field(metadata={"axis": None})
    timestamps: Array = struct.field(metadata={"axis": None})
    returns: Array = struct.field(metadata={"axis": None})
    size: Array = struct.field(metadata={"axis": None})
    episodes: Array = struct.field(metadata={"axis": None})
    exploring: Array = struct.field(metadata={"axis": None})
    key: Key = struct.field(metadata={"axis": None})


class LevelBuffer(Wrapper):
    def __init__(self, env, capacity: int, staging: int, sample: Callable[[Key], PyTree]):
        super().__init__(env)
        self.capacity = capacity
        self.staging = staging
        self.sample = sample

    def init(self, key: Key) -> tuple[LevelBufferState, Timestep]:
        buffer_key, env_key = jax.random.split(key)
        env_state, timestep = self._env.init(env_key)
        slots = self.capacity + self.staging
        template = jax.eval_shape(self.sample, key)
        levels = jax.tree.map(
            lambda leaf: jnp.zeros((slots, *leaf.shape), leaf.dtype), template
        )
        env_state = self._env.update(
            env_state, theta=levels, weights=jnp.zeros(slots, jnp.float32)
        )
        state = LevelBufferState(
            env_state,
            scores=jnp.full(self.capacity, -jnp.inf, jnp.float32),
            timestamps=jnp.zeros(self.capacity, jnp.int32),
            returns=jnp.full(self.capacity, -jnp.inf, jnp.float32),
            size=jnp.int32(0),
            episodes=jnp.int32(0),
            exploring=jnp.bool_(False),
            key=buffer_key,
        )
        return state, timestep

    def step(
        self, key: Key, state: LevelBufferState, action: Array
    ) -> tuple[LevelBufferState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        return state.replace(env_state=env_state), timestep


@struct.dataclass
class GradedState(AlgorithmWrapperState):
    step: Array
    tally: Tally


@dataclass
class Graded(AlgorithmWrapper):
    slots: int
    capacity: int
    gamma: float
    gae_lambda: float
    robust: bool

    def init(self, key: Key, timestep: Timestep) -> GradedState:
        algorithm_state = self.algorithm.init(key, timestep)
        return GradedState(
            algorithm_state, step=algorithm_state.step, tally=blank(self.slots)
        )

    def step(self, state: GradedState, key: Key, timestep: Timestep, temperature=1.0):
        algorithm_state, action, aux = self.algorithm.step(
            state.algorithm_state.replace(step=state.step), key, timestep, temperature
        )
        return state.replace(algorithm_state=algorithm_state), action, aux

    def update(
        self, state: GradedState, key: Key, transitions: Transition
    ) -> GradedState:
        values = transitions.aux["value"]
        advantages, _ = generalized_advantage_estimation(
            transitions, values, values[-1], self.gamma, self.gae_lambda
        )
        counted = tally(
            jax.tree.map(lambda leaf: leaf[1:], transitions),
            values[1:],
            advantages[1:],
            self.slots,
        )
        algorithm_state = state.algorithm_state.replace(step=state.step)
        updated = self.algorithm.update(algorithm_state, key, transitions)
        if not self.robust:
            return state.replace(algorithm_state=updated, tally=counted)

        theta = transitions.second.info["theta"][-1]
        trained = jnp.all((theta >= 0) & (theta < self.capacity))
        updated = updated.replace(
            params=jax.tree.map(
                lambda new, old: jnp.where(trained, new, old),
                updated.params,
                algorithm_state.params,
            ),
            optimizer_state=jax.tree.map(
                lambda new, old: jnp.where(trained, new, old),
                updated.optimizer_state,
                algorithm_state.optimizer_state,
            ),
        )
        return state.replace(algorithm_state=updated, tally=counted)


def rescore(buffer: LevelBufferState, tally: Tally, scored: Array) -> LevelBufferState:
    capacity = buffer.scores.shape[0]
    finished = (tally.episodes[:capacity] > 0) & (jnp.arange(capacity) < buffer.size)
    return buffer.replace(
        scores=jnp.where(finished, scored[:capacity], buffer.scores),
        returns=jnp.where(
            finished,
            jnp.maximum(buffer.returns, tally.returns[:capacity]),
            buffer.returns,
        ),
    )


def draw(
    buffer: LevelBufferState,
    key: Key,
    count: int,
    temperature: float,
    staleness: float,
) -> tuple[LevelBufferState, Array]:
    (capacity,) = buffer.scores.shape

    def pick(buffer, key):
        weights = weigh(
            buffer.scores,
            buffer.timestamps,
            buffer.size,
            buffer.episodes,
            temperature,
            staleness,
        )
        index = jax.random.choice(key, capacity, p=weights).astype(jnp.int32)
        episodes = buffer.episodes + 1
        buffer = buffer.replace(
            timestamps=buffer.timestamps.at[index].set(episodes), episodes=episodes
        )
        return buffer, index

    return jax.lax.scan(pick, buffer, jax.random.split(key, count))


def admit(
    buffer: LevelBufferState,
    levels,
    tally: Tally,
    scored: Array,
    exploring: Array,
    temperature: float,
    staleness: float,
):
    capacity = buffer.scores.shape[0]
    slots = scored.shape[0]

    def insert(carry, source):
        buffer, levels, admitted = carry
        weights = weigh(
            buffer.scores,
            buffer.timestamps,
            buffer.size,
            buffer.episodes,
            temperature,
            staleness,
        )
        target = jnp.where(
            buffer.size < capacity,
            buffer.size,
            jnp.argmin(weights),
        )
        replaced = exploring & (buffer.scores[target] < scored[source])
        episodes = buffer.episodes + exploring.astype(jnp.int32)
        buffer = buffer.replace(
            scores=buffer.scores.at[target].set(
                jnp.where(replaced, scored[source], buffer.scores[target])
            ),
            returns=buffer.returns.at[target].set(
                jnp.where(replaced, tally.returns[source], buffer.returns[target])
            ),
            timestamps=buffer.timestamps.at[target].set(
                jnp.where(replaced, episodes, buffer.timestamps[target])
            ),
            size=jnp.where(
                replaced, jnp.minimum(buffer.size + 1, capacity), buffer.size
            ),
            episodes=episodes,
        )
        levels = jax.tree.map(
            lambda leaf: leaf.at[target].set(
                jnp.where(replaced, leaf[source], leaf[target])
            ),
            levels,
        )
        return (buffer, levels, admitted + replaced.astype(jnp.int32)), None

    (buffer, levels, admitted), _ = jax.lax.scan(
        insert, (buffer, levels, jnp.int32(0)), jnp.arange(capacity, slots)
    )
    return buffer, levels, admitted


def plr(
    algorithm: Algorithm,
    environment: Environment,
    capacity: int,
    replay_probability: float,
    staleness: float,
    temperature: float,
    minimum_fill: float,
    robust: bool,
    gamma: float,
    gae_lambda: float,
    sample: Callable[[Key], PyTree],
    score: Callable[[Tally, Array], Array] = positive_value_loss,
    **kwargs,
) -> tuple[Graded, LevelBuffer, Pit, Lap]:
    staging = environment.num_envs
    slots = capacity + staging
    environment = LevelBuffer(UED(environment), capacity, staging, sample)
    algorithm = Graded(
        algorithm,
        slots=slots,
        capacity=capacity,
        gamma=gamma,
        gae_lambda=gae_lambda,
        robust=robust,
    )

    def pit(state):
        tally = state.algorithm_state.tally
        return state.replace(
            environment_state=within(
                state.environment_state,
                LevelBufferState,
                lambda buffer: refill(buffer, tally),
            )
        )

    def refill(buffer, tally):
        returns = jnp.concatenate(
            [buffer.returns, jnp.full(staging, -jnp.inf, jnp.float32)]
        )
        scored = jnp.where(tally.episodes > 0, score(tally, returns), -jnp.inf)

        buffer = rescore(buffer, tally, scored)
        buffer, levels, admitted = admit(
            buffer,
            buffer.theta,
            tally,
            scored,
            buffer.exploring,
            temperature,
            staleness,
        )

        key, decide_key, draw_key, sample_key = jax.random.split(buffer.key, 4)
        replaying = (buffer.size / capacity >= minimum_fill) & (
            jax.random.uniform(decide_key) < replay_probability
        )

        def replay(buffer, levels):
            buffer, assignment = draw(buffer, draw_key, staging, temperature, staleness)
            return buffer, levels, assignment

        def explore(buffer, levels):
            fresh = jax.vmap(sample)(
                jax.random.split(sample_key, staging)
            )
            levels = jax.tree.map(
                lambda leaf, new: leaf.at[capacity:].set(new), levels, fresh
            )
            return buffer, levels, capacity + jnp.arange(staging, dtype=jnp.int32)

        buffer, levels, assignment = jax.lax.cond(
            replaying, replay, explore, buffer, levels
        )
        level_weights = weigh(
            buffer.scores,
            buffer.timestamps,
            buffer.size,
            buffer.episodes,
            temperature,
            staleness,
        )

        valid = jnp.arange(capacity) < buffer.size
        finished = tally.episodes.sum()
        lox.log(
            {
                "plr/levels/size": buffer.size,
                "plr/levels/admitted": admitted,
                "plr/levels/replaying": replaying.astype(jnp.float32),
                "plr/levels/replayed": tally.episodes[:capacity].sum()
                / jnp.maximum(finished, 1.0),
                "plr/levels/mean_score": jnp.where(valid, buffer.scores, 0.0).sum()
                / jnp.maximum(buffer.size, 1),
                "plr/levels/max_score": jnp.where(
                    buffer.size > 0,
                    jnp.where(valid, buffer.scores, -jnp.inf).max(),
                    jnp.nan,
                ),
                "plr/levels/weighted_score": jnp.where(
                    valid, level_weights * buffer.scores, 0.0
                ).sum(),
                "plr/levels/fresh_score": jnp.where(
                    tally.episodes[capacity:] > 0, scored[capacity:], 0.0
                ).sum()
                / jnp.maximum((tally.episodes[capacity:] > 0).sum(), 1),
            }
        )
        buffer = environment.update(
            buffer.replace(key=key, exploring=~replaying),
            theta=levels,
            assign=assignment,
        )
        return buffer

    return algorithm, environment, pit, lambda state: state
