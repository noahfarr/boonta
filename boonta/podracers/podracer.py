from collections.abc import Callable
from typing import Protocol, Self, TypeVar

import jax
import lox
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from boonta.utils import Key, PyTree, Timestep, Transition
from boonta.utils.typing import EnvState

State = TypeVar("State")


class PodracerState(Protocol):
    @property
    def timestep(self) -> Timestep: ...

    @property
    def environment_state(self) -> EnvState: ...

    @property
    def carry(self) -> PyTree: ...

    @property
    def algorithm_state(self) -> PyTree: ...

    @property
    def pit_state(self) -> PyTree: ...

    @property
    def lap_state(self) -> PyTree: ...

    def replace(self, **changes) -> Self: ...


class Lap:
    def init(self, key: Key, state: PodracerState) -> PyTree:
        return None

    def __call__(self, key: Key, state: PodracerState) -> PodracerState:
        return state


class Pit(Lap):
    def update(
        self, state: PyTree, key: Key, transitions: Transition, algorithm_state: PyTree
    ) -> PyTree:
        return state


class Podracer(Protocol[State]):

    @property
    def batch_size(self) -> int: ...

    def init(self, key: Key) -> State: ...

    def train(self, state: State, key: Key, num_updates: int) -> State: ...

    def evaluate(self, state: State, key: Key, num_steps: int) -> State: ...

    def close(self, state: State) -> State: ...


class sharded_jit:
    def __init__(
        self,
        function: Callable | None = None,
        *,
        inputs=None,
        outputs=None,
        static_argnums: tuple[int, ...] = (),
        donate_argnames: tuple[str, ...] = (),
        strip: str | None = None,
        spool: bool = False,
    ):
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        self.static_argnums = static_argnums
        self.donate_argnames = donate_argnames
        self.strip = strip
        self.spool = spool or strip is not None

    def __call__(self, function: Callable) -> "sharded_jit":
        self.function = function
        return self

    def __set_name__(self, owner, name):
        self.name = name

    def raw(self, podracer) -> Callable:
        return self.function.__get__(podracer, type(podracer))

    def __get__(self, podracer, owner=None):
        if podracer is None:
            return self
        function = self.raw(podracer)
        if self.strip is not None:
            function = lox.strip(function, tags=[self.strip])
        if self.spool:
            function = lox.spool(function)
        options = {}
        if self.inputs is not None:
            options["in_shardings"] = _place(self.inputs, podracer)
        if self.outputs is not None:
            options["out_shardings"] = _place(self.outputs, podracer)
        compiled = jax.jit(
            function,
            static_argnums=self.static_argnums,
            donate_argnames=self.donate_argnames,
            **options,
        )
        podracer.__dict__[self.name] = compiled
        return compiled


def _place(layout, podracer):
    if layout == "state":
        return podracer.shardings
    if layout == "replicated":
        return NamedSharding(podracer.mesh, P())
    return tuple(_place(entry, podracer) for entry in layout)
