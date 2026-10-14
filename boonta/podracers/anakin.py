from dataclasses import dataclass
from functools import partial

import jax
import lox
from boonta.algorithms import Algorithm
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from boonta.utils import Key, PyTree, Timestep, Transition, sharded
from boonta.utils.typing import Environment, EnvState

from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class AnakinConfig:
    num_envs: int
    num_steps: int
    mesh: Mesh = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class AnakinState:
    timestep: Timestep = struct.field(metadata={"axis": "data"})
    environment_state: EnvState = struct.field(metadata={"axis": "data"})
    algorithm_state: PyTree
    pit_state: PyTree = None
    lap_state: PyTree = None


@dataclass
class Anakin:
    config: AnakinConfig
    algorithm: Algorithm
    environment: Environment
    pit: Pit[AnakinState] = Pit()
    lap: Lap[AnakinState] = Lap()

    def rollout(self, state: AnakinState, key, temperature):
        algorithm_key, environment_key, lap_key = jax.random.split(key, 3)

        algorithm_state, action, aux = self.algorithm.step(
            state.algorithm_state, algorithm_key, state.timestep, temperature
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
            + self.config.num_envs * self.environment.num_agents
        )
        state = self.lap(
            lap_key,
            state.replace(
                algorithm_state=algorithm_state,
                timestep=timestep,
                environment_state=environment_state,
            ),
        )
        return state, transition

    def init(self, key: Key) -> AnakinState:
        environment_key, algorithm_key, pit_key, lap_key = jax.random.split(key, 4)
        pit_key, call_key = jax.random.split(pit_key)
        lap_key, step_key = jax.random.split(lap_key)
        environment_state, timestep = self.environment.init(environment_key)
        algorithm_state = self.algorithm.init(algorithm_key, timestep)
        return self.pit(
            call_key,
            self.lap(
                step_key,
                AnakinState(
                    timestep=timestep,
                    environment_state=environment_state,
                    algorithm_state=algorithm_state,
                    pit_state=self.pit.init(pit_key),
                    lap_state=self.lap.init(lap_key),
                ),
            ),
        )

    @property
    def batch_size(self) -> int:
        return (
            self.config.num_envs * self.config.num_steps * self.environment.num_agents
        )

    def train(self, state: AnakinState, key: Key, num_updates: int) -> PyTree:
        def step(state: AnakinState, key):
            rollout_key, update_key, pit_key, call_key = jax.random.split(key, 4)

            rollout_keys = jax.random.split(rollout_key, self.config.num_steps)
            state, transitions = jax.lax.scan(
                partial(self.rollout, temperature=1.0), state, rollout_keys
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
        return state

    def evaluate(self, state: AnakinState, key: Key, num_steps: int) -> PyTree:
        reset_key, rollout_key, lap_key, call_key = jax.random.split(key, 4)
        rollout_keys = jax.random.split(rollout_key, num_steps)
        environment_state, timestep = self.environment.init(reset_key)
        state = self.pit(
            call_key,
            self.lap(
                lap_key,
                state.replace(
                    timestep=timestep,
                    environment_state=environment_state,
                ),
            ),
        )
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0), state, rollout_keys
        )
        self.environment.close(state.environment_state)
        return state

    def close(self, state: AnakinState) -> AnakinState:
        self.environment.close(state.environment_state)
        return state


def make(
    config: AnakinConfig,
    algorithm: Algorithm,
    environment: Environment,
    pit: Pit[AnakinState] = Pit(),
    lap: Lap[AnakinState] = Lap(),
) -> Anakin:
    podracer = Anakin(config, algorithm, environment, pit, lap)

    initialize, close = podracer.init, podracer.close
    train = lox.spool(lox.strip(podracer.train, tags=["evaluation"]))
    evaluate = lox.spool(lox.strip(podracer.evaluate, tags=["training"]))

    replicated = NamedSharding(config.mesh, P())
    axes = sharded(jax.eval_shape(initialize, jax.random.key(0)))

    shardings = jax.tree.map(
        lambda axis: NamedSharding(config.mesh, P(axis)) if axis else replicated, axes
    )

    podracer.init = jax.jit(initialize, out_shardings=shardings)
    podracer.train = jax.jit(
        train,
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
    podracer.close = jax.jit(close, in_shardings=(shardings,))
    return podracer
