import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import lox
from boonta.algorithms import Algorithm
from boonta.datasets import Dataset, DatasetState
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from boonta.utils import Key, PyTree, Timestep, sharded
from boonta.utils.typing import Environment, EnvState

from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class QuadinarosConfig:
    num_envs: int
    batch_shape: tuple[int, ...]
    mesh: Mesh = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class QuadinarosState:
    timestep: Timestep = struct.field(metadata={"axis": "data"})
    environment_state: EnvState = struct.field(metadata={"axis": "data"})
    dataset_state: DatasetState = struct.field(metadata={"axis": "data"})
    algorithm_state: PyTree


@dataclass
class Quadinaros:
    config: QuadinarosConfig
    algorithm: Algorithm
    environment: Environment
    dataset: Dataset
    pit: Pit[QuadinarosState] = lambda state, transitions: state
    lap: Lap[QuadinarosState] = lambda state: state

    def step(self, state: QuadinarosState, key):
        update_key, sample_key = jax.random.split(key)
        batch = self.dataset.sample(
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
        return self.pit(state.replace(algorithm_state=algorithm_state), batch), None

    def rollout(self, state: QuadinarosState, key, temperature):
        algorithm_key, environment_key = jax.random.split(key)

        algorithm_state, action, aux = self.algorithm.step(
            state.algorithm_state, algorithm_key, state.timestep, temperature
        )
        environment_state, timestep = self.environment.step(
            environment_key, state.environment_state, action
        )

        algorithm_state = algorithm_state.replace(
            step=algorithm_state.step
            + self.config.num_envs * self.environment.num_agents
        )
        state = self.lap(
            state.replace(
                algorithm_state=algorithm_state,
                timestep=timestep,
                environment_state=environment_state,
            )
        )
        return state, None

    def init(self, key: Key) -> QuadinarosState:
        environment_key, algorithm_key = jax.random.split(key)
        environment_state, timestep = self.environment.init(environment_key)
        algorithm_state = self.algorithm.init(algorithm_key, timestep)
        return self.pit(
            self.lap(
                QuadinarosState(
                    timestep=timestep,
                    environment_state=environment_state,
                    dataset_state=self.dataset.init(),
                    algorithm_state=algorithm_state,
                )
            ),
            None,
        )

    @property
    def batch_size(self) -> int:
        return math.prod(self.config.batch_shape)

    def fit(self, state: QuadinarosState, key: Key, num_updates: int) -> PyTree:
        keys = jax.random.split(key, num_updates)
        state, _ = jax.lax.scan(self.step, state, keys)
        return state

    def train(self, state: QuadinarosState, key: Key, num_updates: int) -> PyTree:
        sharding = jax.tree.map(lambda leaf: leaf.sharding, state.dataset_state)
        dataset_state = self.dataset.update(
            state.dataset_state, jax.random.fold_in(key, 1), sharding
        )
        return self.fit(state.replace(dataset_state=dataset_state), key, num_updates)

    def evaluate(self, state: QuadinarosState, key: Key, num_steps: int) -> PyTree:
        reset_key, rollout_key = jax.random.split(key)
        rollout_keys = jax.random.split(rollout_key, num_steps)
        environment_state, timestep = self.environment.init(reset_key)
        state = self.pit(
            self.lap(
                state.replace(
                    timestep=timestep,
                    environment_state=environment_state,
                )
            ),
            None,
        )
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0), state, rollout_keys
        )
        self.environment.close(state.environment_state)
        return state

    def close(self, state: QuadinarosState) -> QuadinarosState:
        self.environment.close(state.environment_state)
        self.dataset.close()
        return state


def untangle(tree: PyTree) -> PyTree:
    seen = set()

    def own(leaf):
        if id(leaf) in seen:
            return jnp.copy(leaf)
        seen.add(id(leaf))
        return leaf

    return jax.tree.map(own, tree)


def make(
    config: QuadinarosConfig,
    algorithm: Algorithm,
    environment: Environment,
    dataset: Dataset,
    pit: Pit[QuadinarosState] = lambda state, transitions: state,
    lap: Lap[QuadinarosState] = lambda state: state,
) -> Quadinaros:
    config = config.replace(batch_shape=tuple(config.batch_shape))
    podracer = Quadinaros(config, algorithm, environment, dataset, pit, lap)

    init = podracer.init
    fit = lox.spool(lox.strip(podracer.fit, tags=["evaluation"]))
    evaluate = lox.spool(lox.strip(podracer.evaluate, tags=["training"]))

    replicated = NamedSharding(config.mesh, P())
    axes = sharded(jax.eval_shape(init, jax.random.key(0)))

    shardings = jax.tree.map(
        lambda axis: NamedSharding(config.mesh, P(axis)) if axis else replicated, axes
    )

    podracer.init = lambda key: jax.device_put(untangle(init(key)), shardings)
    podracer.fit = jax.jit(
        fit,
        static_argnums=(2,),
        donate_argnames=("state",),
        in_shardings=(shardings, replicated),
        out_shardings=(shardings, replicated),
    )
    podracer.evaluate = jax.jit(
        evaluate,
        static_argnums=(2,),
        in_shardings=(shardings, replicated),
        out_shardings=(shardings, replicated),
    )
    return podracer
