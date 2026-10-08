from dataclasses import dataclass
from typing import Generic, TypeVar

import jax

from boonta.utils.typing import Array, Key

from .planner import Planner, RolloutFn, SampleFn

State = TypeVar("State")


@dataclass
class MPC(Generic[State]):
    planner: Planner[State]

    def init(self, key: Key) -> State:
        return self.planner.init(key)

    def step(
        self,
        key: Key,
        state: State,
        rollout_fn: RolloutFn,
        sample_fn: SampleFn,
        done: Array,
        temperature: float,
    ) -> tuple[State, Array]:
        plan_key, sample_key = jax.random.split(key)

        state = self.planner.reset(state, done)
        state = self.planner.plan(plan_key, state, rollout_fn, sample_fn)
        action = self.planner.sample(sample_key, state, temperature)
        return state, action
