from dataclasses import dataclass, fields, replace
from typing import Any

import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, Transition, canonicalize_dtype

from ..algorithm import Algorithm
from .wrapper import Wrapper, WrapperState


@struct.dataclass(frozen=True)
class InjectedState(WrapperState):
    algorithm_state: Any
    values: dict
    step: Array


@dataclass
class Injected(Wrapper):
    names: tuple[str, ...] = ()

    def __post_init__(self):
        self.names = tuple(self.names)
        if self.names and "cfg" not in {field.name for field in fields(self.algorithm)}:
            raise ValueError(
                f"{type(self.algorithm).__name__} has no `cfg` to inject "
                f"{list(self.names)} into."
            )
        if not self.names:
            return
        kinds = {field.name: field.type for field in fields(self.algorithm.cfg)}
        unknown = [name for name in self.names if name not in kinds]
        if unknown:
            raise ValueError(
                f"{type(self.algorithm.cfg).__name__} has no fields {unknown}; "
                f"it has {sorted(kinds)}."
            )
        rejected = [name for name in self.names if kinds[name] not in (float, "float")]
        if rejected:
            raise ValueError(
                f"only float fields can be injected, but "
                f"{type(self.algorithm.cfg).__name__} declares "
                f"{ {name: kinds[name] for name in rejected} }. Integers and "
                f"booleans set shapes or Python control flow and must stay static."
            )

    def defaults(self) -> dict:
        return {
            name: jnp.asarray(getattr(self.algorithm.cfg, name), jnp.float32)
            for name in self.names
        }

    def rebuild(self, values: dict) -> Algorithm:
        if not self.names:
            return self.algorithm
        return replace(self.algorithm, cfg=replace(self.algorithm.cfg, **values))

    def inner(self, state: InjectedState) -> PyTree:
        return state.algorithm_state.replace(step=state.step)

    def init(self, key: Key, timestep: Timestep) -> InjectedState:
        values = self.defaults()
        algorithm_state = self.rebuild(values).init(key, timestep)
        return InjectedState(
            algorithm_state=algorithm_state,
            values=values,
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )

    def step(
        self, state: InjectedState, key: Key, timestep: Timestep, temperature=1.0
    ) -> tuple[InjectedState, Array, PyTree]:
        algorithm_state, action, aux = self.rebuild(state.values).step(
            self.inner(state), key, timestep, temperature
        )
        return state.replace(algorithm_state=algorithm_state), action, aux

    def update(
        self, state: InjectedState, key: Key, transitions: Transition
    ) -> InjectedState:
        algorithm_state = self.rebuild(state.values).update(
            self.inner(state), key, transitions
        )
        return state.replace(algorithm_state=algorithm_state)
