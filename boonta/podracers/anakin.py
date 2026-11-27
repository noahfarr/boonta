from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import lox
from boonta.algorithms import Algorithm
from flax import struct
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from boonta.utils import Key, PyTree, Timestep, Transition, pin
from boonta.utils.typing import Environment, EnvState

from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class AnakinConfig:
    num_steps: int
    mesh: Mesh = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class AnakinState:
    timestep: Timestep = struct.field(metadata={"axis": "batch"})
    environment_state: EnvState = struct.field(metadata={"axis": "batch"})
    carry: PyTree = struct.field(metadata={"axis": "batch"})
    algorithm_state: PyTree
    pit_state: PyTree = None
    lap_state: PyTree = None


@dataclass(eq=False)
class Anakin:
    config: AnakinConfig
    algorithm: Algorithm
    environment: Environment
    pit: Pit = field(default_factory=Pit)
    lap: Lap = field(default_factory=Lap)
    partition_rules: Sequence[tuple[str, P]] = ()

    def pin(self, state: AnakinState) -> AnakinState:
        return pin(state, self.config.mesh, self.partition_rules)

    def rollout(self, state: AnakinState, key, temperature):
        algorithm_key, environment_key, lap_key = jax.random.split(key, 3)

        algorithm_state, carry, action, aux = self.algorithm.step(
            state.algorithm_state, state.carry, algorithm_key, state.timestep, temperature
        )
        environment_state, timestep = self.environment.step(
            environment_key, state.environment_state, action
        )

        transition = Transition(
            first=state.timestep,
            second=timestep,
            aux=aux,
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
        return state, transition

    @partial(jax.jit, static_argnums=0)
    def init(self, key: Key) -> AnakinState:
        environment_key, algorithm_key, pit_key, lap_key = jax.random.split(key, 4)
        pit_key, call_key = jax.random.split(pit_key)
        lap_key, step_key = jax.random.split(lap_key)
        environment_state, timestep = self.environment.init(environment_key)
        algorithm_state, carry = self.algorithm.init(algorithm_key, timestep)
        state = AnakinState(
            timestep=timestep,
            environment_state=environment_state,
            carry=carry,
            algorithm_state=algorithm_state,
        )
        state = state.replace(
            pit_state=self.pit.init(pit_key, state),
            lap_state=self.lap.init(lap_key, state),
        )
        return self.pin(self.pit(call_key, self.lap(step_key, state)))

    @property
    def batch_size(self) -> int:
        return (
            self.environment.num_envs
            * self.config.num_steps
            * self.environment.num_agents
        )

    @partial(jax.jit, static_argnums=(0, 3), donate_argnames="state")
    @lox.spool
    @partial(lox.strip, tags=["evaluation"])
    def train(self, state: AnakinState, key: Key, num_updates: int) -> PyTree:
        def step(state: AnakinState, key):
            rollout_key, update_key, pit_key, call_key = jax.random.split(key, 4)

            state = self.pin(state)
            rollout_keys = jax.random.split(rollout_key, self.config.num_steps)
            initial_carry = state.carry
            state, transitions = jax.lax.scan(
                partial(self.rollout, temperature=1.0), state, rollout_keys
            )
            transitions = transitions.replace(
                carry=jax.tree.map(lambda *carry: jnp.stack(carry), initial_carry, state.carry)
            )
            algorithm_state = self.algorithm.update(
                state.algorithm_state, update_key, transitions
            )
            pit_state = self.pit.update(
                state.pit_state, pit_key, transitions, algorithm_state
            )
            state = state.replace(algorithm_state=algorithm_state, pit_state=pit_state)
            return self.pit(call_key, state), None

        keys = jax.random.split(key, num_updates)
        state, _ = jax.lax.scan(step, state, keys)
        return self.pin(state)

    @partial(jax.jit, static_argnums=(0, 3))
    @lox.spool
    @partial(lox.strip, tags=["training"])
    def evaluate(self, state: AnakinState, key: Key, num_steps: int) -> PyTree:
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

    @partial(jax.jit, static_argnums=0)
    def close(self, state: AnakinState) -> AnakinState:
        self.environment.close(state.environment_state)
        return state

