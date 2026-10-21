from dataclasses import dataclass, field, replace

import jax
from flax import struct
from boonta.utils import Array, Key, PyTree, Timestep, Transition, conditional_update, place

from ..algorithm import Algorithm, Carries
from ..auxiliary_losses import Anchor
from .population import Population, PopulationState


@struct.dataclass(frozen=True)
class RosterState(PopulationState):
    initial_states: tuple = ()


@dataclass
class Roster(Population):
    initial_params: dict[int, PyTree] = field(default_factory=dict)
    kl_coefficients: tuple[float, ...] = ()

    def __post_init__(self):
        if any(coefficient > 0.0 for coefficient in self.kl_coefficients):
            assert hasattr(self.algorithm, "auxiliary_losses"), (
                f"{type(self.algorithm).__name__} takes no auxiliary losses, but some "
                f"learners set kl_coefficient > 0 to stay near their initial "
                f"parameters. Use kl_coefficient=0.0 for every learner, or an oracle "
                f"that can be anchored."
            )

    def init(self, key: Key, timestep: Timestep) -> tuple[RosterState, PyTree]:
        state, carry = super().init(key, timestep)
        algorithm_states = tuple(
            algorithm_state.replace(
                params=self.initial_params.get(index, algorithm_state.params)
            )
            for index, algorithm_state in enumerate(state.algorithm_states)
        )
        state = RosterState(
            algorithm_states=algorithm_states,
            step=state.step,
            initial_states=algorithm_states,
        )
        return state, carry

    def anchored(self, state: RosterState, index: int) -> Algorithm:
        coefficient = (self.kl_coefficients or (0.0,) * self.count)[index]
        if coefficient <= 0.0:
            return self.algorithm
        return replace(
            self.algorithm,
            auxiliary_losses=(
                *self.algorithm.auxiliary_losses,
                Anchor(state.initial_states[index].params, coefficient),
            ),
        )

    def update(
        self, state: RosterState, key: Key, transitions: Transition, carries: Carries
    ) -> RosterState:
        state = self.synchronize(state)
        for index in range(self.count):
            state = self.respond(
                state,
                index,
                jax.random.fold_in(key, index),
                self.portion(transitions, index),
                self.share(carries, index),
                self.anchored(state, index),
            )
        return state

    def restart(
        self, state: RosterState, index: int, when: Array, params: PyTree = None
    ) -> RosterState:
        initial = state.initial_states[index]
        if params is not None:
            initial = initial.replace(params=params)
        current = state.algorithm_states[index]
        return state.replace(
            algorithm_states=place(
                state.algorithm_states,
                index,
                conditional_update(initial, current, when),
            )
        )
