from collections.abc import Callable
from typing import Protocol, TypeVar

from boonta.utils.typing import Array, Key

State = TypeVar("State")

RolloutFn = Callable[..., Array]
SampleFn = Callable[..., Array]


class Planner(Protocol[State]):
    def init(self, key: Key) -> State: ...

    def reset(self, state: State, done: Array) -> State: ...

    def plan(
        self, key: Key, state: State, rollout_fn: RolloutFn, sample_fn: SampleFn
    ) -> State: ...

    def sample(self, key: Key, state: State, temperature: float) -> Array: ...
