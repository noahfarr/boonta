from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.environments.wrappers import UED
from boonta.environments.wrappers.ued import UEDState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Trajectory, Transition
from boonta.utils.typing import Environment


@struct.dataclass
class TaskPool:
    levels: PyTree
    outcomes: Array
    num_outcomes: Array
    cursor: Array


@dataclass
class SGS(Pit):
    environment: UED
    pool_size: int
    num_envs: int
    history_length: int
    target_success: float
    concentration: float
    epsilon: float
    temperature: float
    success_threshold: float
    generate_level: Callable[[Key], PyTree]

    def success_rates(self, pool: TaskPool) -> Array:
        return pool.outcomes.sum(axis=-1) / jnp.maximum(pool.num_outcomes, 1)

    def sampling_weights(self, pool: TaskPool) -> Array:
        rates = self.success_rates(pool)
        kernel = (rates + self.epsilon) ** (self.concentration * self.target_success) * (
            1.0 - rates + self.epsilon
        ) ** (self.concentration * (1.0 - self.target_success))
        logits = jnp.log(jnp.maximum(kernel, self.epsilon) + self.epsilon)
        return jax.nn.softmax(logits / max(self.temperature, 1.0))

    def count_outcomes(self, transitions: Transition) -> tuple[Array, Array]:
        second = transitions.second
        theta = second.info["theta"]
        index = jnp.where(second.done & (theta >= 0), theta, self.pool_size).reshape(-1)
        succeeded = second.terminated & (second.reward > self.success_threshold)
        succeeded = succeeded.astype(jnp.int32).reshape(-1)
        empty = jnp.zeros(self.pool_size, jnp.int32)
        episodes = empty.at[index].add(1, mode="drop")
        successes = empty.at[index].add(succeeded, mode="drop")
        return episodes, successes

    def record(self, pool: TaskPool, episodes: Array, successes: Array) -> TaskPool:
        kept = jnp.minimum(episodes, self.history_length)
        kept_successes = jnp.round(successes * kept / jnp.maximum(episodes, 1))
        offset = (jnp.arange(self.history_length) - pool.cursor[:, None]) % self.history_length
        written = offset < kept[:, None]
        outcomes = jnp.where(
            written, (offset < kept_successes[:, None]).astype(jnp.float32), pool.outcomes
        )
        return pool.replace(
            outcomes=outcomes,
            num_outcomes=jnp.minimum(pool.num_outcomes + kept, self.history_length),
            cursor=(pool.cursor + kept) % self.history_length,
        )

    def init(self, key: Key, state: PyTree) -> TaskPool:
        return TaskPool(
            levels=jax.vmap(self.generate_level)(jax.random.split(key, self.pool_size)),
            outcomes=jnp.zeros((self.pool_size, self.history_length), jnp.float32),
            num_outcomes=jnp.zeros(self.pool_size, jnp.int32),
            cursor=jnp.zeros(self.pool_size, jnp.int32),
        )

    def update(
        self,
        pool: TaskPool,
        key: Key,
        trajectory: Trajectory,
        algorithm_state: PyTree,
    ) -> TaskPool:
        transitions = trajectory.transitions
        episodes, successes = self.count_outcomes(transitions)
        pool = self.record(pool, episodes, successes)
        self.report(pool, episodes, successes)
        return pool

    def report(self, pool: TaskPool, episodes: Array, successes: Array) -> None:
        tried = pool.num_outcomes > 0
        rates = self.success_rates(pool)
        num_tried = jnp.maximum(jnp.sum(tried, dtype=jnp.float32), 1.0)
        weights = self.sampling_weights(pool)
        lox.log(
            {
                "pool/num_tried": jnp.sum(tried, dtype=jnp.float32),
                "pool/success_rate": jnp.sum(jnp.where(tried, rates, 0.0)) / num_tried,
                "mu/success_rate": jnp.sum(weights * rates),
                "mu/episode_success": jnp.sum(successes) / jnp.maximum(jnp.sum(episodes), 1),
            }
        )

    def __call__(self, key: Key, state):
        pool = state.pit_state

        def assign(node):
            if isinstance(node, UEDState):
                return self.environment.update(
                    node,
                    key,
                    theta=pool.levels,
                    weights=self.sampling_weights(pool),
                    restart=jnp.any(node.playing < 0),
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
    pool_size: int,
    generate_level: Callable[[Key], PyTree],
    history_length: int = 100,
    target_success: float = 0.5,
    concentration: float = 1.0,
    epsilon: float = 1e-4,
    temperature: float = 2.0,
    success_threshold: float = 0.0,
    **kwargs,
) -> tuple[Algorithm, UED, SGS, Lap | None]:
    pit = SGS(
        UED(environment),
        pool_size=pool_size,
        num_envs=environment.num_envs,
        history_length=history_length,
        target_success=target_success,
        concentration=concentration,
        epsilon=epsilon,
        temperature=temperature,
        success_threshold=success_threshold,
        generate_level=generate_level,
    )
    return algorithm, pit.environment, pit, None
