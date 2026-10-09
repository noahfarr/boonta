from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp

from boonta.utils import Array, Key, PyTree, Timestep

from .spaces import Space

State = TypeVar("State", bound=PyTree)


class Environment(ABC, Generic[State]):

    num_agents: int = 1

    @abstractmethod
    def init(self, key: Key) -> tuple[State, Timestep]: ...

    @abstractmethod
    def step(self, key: Key, state: State, action: Array) -> tuple[State, Timestep]: ...

    def update(self, state: State, **kwargs) -> State:
        return state

    @abstractmethod
    def observation_space(self) -> Space: ...

    @abstractmethod
    def action_space(self) -> Space: ...

    def action_mask(self, state: State) -> Array | None:
        space = self.action_space()
        if not jnp.issubdtype(space.dtype, jnp.integer):
            return None
        return jnp.ones((*space.shape, space.num_actions), dtype=bool)

    def observe(self, state: State) -> PyTree:
        raise NotImplementedError

    def time_limit(self) -> int:
        raise NotImplementedError

    def render(self, state: State):
        raise NotImplementedError

    def close(self, state: State) -> None:
        pass
