from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from boonta.utils import Array, Key, PyTree, Timestep, Transition, concatenate, place, take

from ..algorithm import Algorithm
from .wrapper import Wrapper


@struct.dataclass(frozen=True)
class EnsembleState:
    algorithm_states: tuple
    step: Array

    @property
    def params(self) -> PyTree:
        return {
            f"role:{index}": algorithm_state.params
            for index, algorithm_state in enumerate(self.algorithm_states)
        }


@dataclass
class Ensemble(Wrapper):
    count: int

    def init(self, key: Key, timestep: Timestep) -> EnsembleState:
        slots = timestep.terminated.shape[0]
        assert slots % self.count == 0, (
            f"the environment's {slots} slots do not divide evenly among "
            f"{self.count} copies: each would be floored to {slots // self.count}, "
            f"and the actions concatenated back together would be "
            f"{self.count * (slots // self.count)} wide. "
            f"Pick a multiple of {self.count}."
        )
        return EnsembleState(
            algorithm_states=tuple(
                self.algorithm.init(
                    jax.random.fold_in(key, index), take(timestep, index, self.count)
                )
                for index in range(self.count)
            ),
            step=jnp.array(0),
        )

    def step(
        self, state: EnsembleState, key: Key, timestep: Timestep, temperature=1.0
    ) -> tuple[EnsembleState, Array, PyTree]:
        algorithm_states, actions, auxes = [], [], []
        for index, algorithm_state in enumerate(state.algorithm_states):
            view = take(timestep, index, self.count)
            algorithm_state, action, aux = self.algorithm.step(
                algorithm_state, jax.random.fold_in(key, index), view, temperature
            )
            algorithm_states.append(
                algorithm_state.replace(step=algorithm_state.step + action.shape[0])
            )
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
        state: EnsembleState,
        index: int,
        key: Key,
        transitions: Transition,
        algorithm: Algorithm,
    ) -> EnsembleState:
        algorithm_state = algorithm.update(
            state.algorithm_states[index], key, transitions
        )
        return state.replace(
            algorithm_states=place(state.algorithm_states, index, algorithm_state)
        )

    def update(
        self, state: EnsembleState, key: Key, transitions: Transition
    ) -> EnsembleState:
        for index in range(self.count):
            state = self.respond(
                state,
                index,
                jax.random.fold_in(key, index),
                self.portion(transitions, index),
                self.algorithm,
            )
        return state
