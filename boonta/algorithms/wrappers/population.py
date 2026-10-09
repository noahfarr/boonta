from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from boonta.utils import (Array, Key, PyTree, Timestep, Transition,
                          canonicalize_dtype, concatenate, place, take)

from ..algorithm import Algorithm
from .wrapper import Wrapper


@struct.dataclass(frozen=True)
class PopulationState:
    algorithm_states: tuple
    step: Array

    @property
    def params(self) -> PyTree:
        return {
            f"role:{index}": algorithm_state.params
            for index, algorithm_state in enumerate(self.algorithm_states)
        }


@dataclass
class Population(Wrapper):
    count: int

    def init(self, key: Key, timestep: Timestep) -> PopulationState:
        slots = timestep.terminated.shape[0]
        assert slots % self.count == 0, (
            f"the environment's {slots} slots do not divide evenly among "
            f"{self.count} copies: each would be floored to {slots // self.count}, "
            f"and the actions concatenated back together would be "
            f"{self.count * (slots // self.count)} wide. "
            f"Pick a multiple of {self.count}."
        )
        return PopulationState(
            algorithm_states=tuple(
                self.algorithm.init(
                    jax.random.fold_in(key, index), take(timestep, index, self.count)
                )
                for index in range(self.count)
            ),
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )

    def synchronize(self, state: PopulationState) -> PopulationState:
        return state.replace(
            algorithm_states=tuple(
                algorithm_state.replace(step=state.step)
                for algorithm_state in state.algorithm_states
            )
        )

    def step(
        self, state: PopulationState, key: Key, timestep: Timestep, temperature=1.0
    ) -> tuple[PopulationState, Array, PyTree]:
        state = self.synchronize(state)
        algorithm_states, actions, auxes = [], [], []
        for index, algorithm_state in enumerate(state.algorithm_states):
            view = take(timestep, index, self.count)
            algorithm_state, action, aux = self.algorithm.step(
                algorithm_state, jax.random.fold_in(key, index), view, temperature
            )
            algorithm_states.append(algorithm_state)
            actions.append(action)
            auxes.append(aux)
        return (
            state.replace(algorithm_states=tuple(algorithm_states)),
            concatenate(actions),
            concatenate(auxes),
        )

    def portion(self, transitions: Transition, index: int) -> Transition:
        return take(transitions, index, self.count, axis=1)

    def respond(
        self,
        state: PopulationState,
        index: int,
        key: Key,
        transitions: Transition,
        algorithm: Algorithm,
    ) -> PopulationState:
        algorithm_state = algorithm.update(
            state.algorithm_states[index], key, transitions
        )
        return state.replace(
            algorithm_states=place(state.algorithm_states, index, algorithm_state)
        )

    def update(
        self, state: PopulationState, key: Key, transitions: Transition
    ) -> PopulationState:
        state = self.synchronize(state)
        for index in range(self.count):
            state = self.respond(
                state,
                index,
                jax.random.fold_in(key, index),
                self.portion(transitions, index),
                self.algorithm,
            )
        return state
