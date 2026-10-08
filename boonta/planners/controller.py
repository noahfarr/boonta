from typing import Protocol, TypeVar

from boonta.utils.typing import Array, Key

from .planner import RolloutFn, SampleFn

State = TypeVar("State")


class Controller(Protocol[State]):
    def init(self, key: Key) -> State: ...

    def step(
        self,
        key: Key,
        state: State,
        rollout_fn: RolloutFn,
        sample_fn: SampleFn,
        done: Array,
        temperature: float,
    ) -> tuple[State, Array]: ...
