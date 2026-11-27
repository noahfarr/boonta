import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import jax
import lox
from boonta.algorithms import Algorithm
from boonta.datasets import Dataset, DatasetState
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from boonta.utils import Key, PyTree, Timestep, pin, put
from boonta.utils.typing import Environment, EnvState

from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class QuadinarosConfig:
    batch_shape: tuple[int, ...]
    mesh: Mesh = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class QuadinarosState:
    timestep: Timestep = struct.field(metadata={"axis": "batch"})
    environment_state: EnvState = struct.field(metadata={"axis": "batch"})
    dataset_state: DatasetState = struct.field(metadata={"axis": "batch"})
    carry: PyTree = struct.field(metadata={"axis": "batch"})
    algorithm_state: PyTree
    pit_state: PyTree = None
    lap_state: PyTree = None


@dataclass(eq=False)
class Quadinaros:
    config: QuadinarosConfig
    algorithm: Algorithm
    environment: Environment
    dataset: Dataset
    pit: Pit = field(default_factory=Pit)
    lap: Lap = field(default_factory=Lap)
    partition_rules: Sequence[tuple[str, P]] = ()

    def __post_init__(self):
        self.config = self.config.replace(batch_shape=tuple(self.config.batch_shape))

    def pin(self, state: QuadinarosState) -> QuadinarosState:
        return pin(state, self.config.mesh, self.partition_rules)

    def init(self, key: Key) -> QuadinarosState:
        return put(self.start(key), self.config.mesh, self.partition_rules, may_alias=False)

    def step(self, state: QuadinarosState, key):
        update_key, sample_key, pit_key, call_key = jax.random.split(key, 4)
        state = self.pin(state)
        dataset_state, batch = self.dataset.sample(
            state.dataset_state, sample_key, self.config.batch_shape
        )
        batch = jax.lax.with_sharding_constraint(
            batch, NamedSharding(self.config.mesh, P("data"))
        )
        algorithm_state = self.algorithm.update(
            state.algorithm_state, update_key, batch
        )
        algorithm_state = algorithm_state.replace(
            step=algorithm_state.step + self.batch_size
        )
        pit_state = self.pit.update(state.pit_state, pit_key, batch, algorithm_state)
        state = state.replace(
            algorithm_state=algorithm_state, dataset_state=dataset_state, pit_state=pit_state
        )
        return self.pit(call_key, state), None

    def rollout(self, state: QuadinarosState, key, temperature):
        algorithm_key, environment_key, lap_key = jax.random.split(key, 3)

        algorithm_state, carry, action, aux = self.algorithm.step(
            state.algorithm_state, state.carry, algorithm_key, state.timestep, temperature
        )
        environment_state, timestep = self.environment.step(
            environment_key, state.environment_state, action
        )

        algorithm_state = algorithm_state.replace(
            step=algorithm_state.step
            + self.environment.num_envs * self.environment.num_agents
        )
        state = self.lap(
            lap_key,
            state.replace(
                algorithm_state=algorithm_state,
                carry=carry,
                timestep=timestep,
                environment_state=environment_state,
            ),
        )
        return state, None

    def start(self, key: Key) -> QuadinarosState:
        environment_key, algorithm_key, pit_key, lap_key = jax.random.split(key, 4)
        pit_key, call_key = jax.random.split(pit_key)
        lap_key, step_key = jax.random.split(lap_key)
        environment_state, timestep = self.environment.init(environment_key)
        algorithm_state, carry = self.algorithm.init(algorithm_key, timestep)
        state = QuadinarosState(
            timestep=timestep,
            environment_state=environment_state,
            dataset_state=self.dataset.init(),
            carry=carry,
            algorithm_state=algorithm_state,
        )
        state = state.replace(
            pit_state=self.pit.init(pit_key, state),
            lap_state=self.lap.init(lap_key, state),
        )
        return self.pit(call_key, self.lap(step_key, state))

    @property
    def batch_size(self) -> int:
        return math.prod(self.config.batch_shape)

    @partial(jax.jit, static_argnums=(0, 3), donate_argnames="state")
    @lox.spool
    @partial(lox.strip, tags=["evaluation"])
    def fit(self, state: QuadinarosState, key: Key, num_updates: int) -> PyTree:
        keys = jax.random.split(key, num_updates)
        state, _ = jax.lax.scan(self.step, state, keys)
        return self.pin(state)

    def train(self, state: QuadinarosState, key: Key, num_updates: int) -> PyTree:
        sharding = jax.tree.map(lambda leaf: leaf.sharding, state.dataset_state)
        dataset_state = self.dataset.update(
            state.dataset_state, jax.random.fold_in(key, 1), sharding
        )
        return self.fit(state.replace(dataset_state=dataset_state), key, num_updates)

    @partial(jax.jit, static_argnums=(0, 3))
    @lox.spool
    @partial(lox.strip, tags=["training"])
    def evaluate(self, state: QuadinarosState, key: Key, num_steps: int) -> PyTree:
        reset_key, rollout_key, lap_key, call_key = jax.random.split(key, 4)
        rollout_keys = jax.random.split(rollout_key, num_steps)
        environment_state, timestep = self.environment.init(reset_key)
        _, carry = self.algorithm.init(reset_key, timestep)
        state = self.pit(
            call_key,
            self.lap(
                lap_key,
                state.replace(
                    timestep=timestep,
                    environment_state=environment_state,
                    carry=carry,
                ),
            ),
        )
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0), self.pin(state), rollout_keys
        )
        self.environment.close(state.environment_state)
        return self.pin(state)

    def close(self, state: QuadinarosState) -> QuadinarosState:
        self.environment.close(state.environment_state)
        self.dataset.close()
        return state

