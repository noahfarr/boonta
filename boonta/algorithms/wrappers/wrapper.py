from dataclasses import dataclass
from typing import Any

from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, Transition

from ..algorithm import Algorithm


@struct.dataclass
class WrapperState:
    algorithm_state: Any

    def __getattr__(self, name):
        return getattr(self.algorithm_state, name)

    @property
    def unwrapped(self):
        return getattr(self.algorithm_state, "unwrapped", self.algorithm_state)


@dataclass
class Wrapper:
    algorithm: Algorithm

    def __getattr__(self, name):
        if name == "algorithm":
            raise AttributeError(name)
        return getattr(self.algorithm, name)

    def init(self, key: Key, timestep: Timestep):
        return self.algorithm.init(key, timestep)

    def step(
        self, state, key: Key, timestep: Timestep, temperature: float = 1.0
    ) -> tuple[Any, Array, PyTree]:
        return self.algorithm.step(state, key, timestep, temperature)

    def update(self, state, key: Key, transitions: Transition):
        return self.algorithm.update(state, key, transitions)

    def wraps(self, cls: type) -> bool:
        algorithm = self.algorithm
        while isinstance(algorithm, Wrapper):
            if isinstance(algorithm, cls):
                return True
            algorithm = algorithm.algorithm
        return isinstance(algorithm, cls)
