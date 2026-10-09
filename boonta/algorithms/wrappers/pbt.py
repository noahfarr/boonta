from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import optax
from flax import struct

from boonta.utils import Array, Key, Timestep, Transition, canonicalize_dtype

from ..algorithm import Algorithm
from .injected import Injected
from .population import Population, PopulationState
from .wrapper import Wrapper, WrapperState

MULTIPLIER = "learning_rate_multiplier"


def scale_by_multiplier(multiplier: float) -> optax.GradientTransformation:
    return optax.scale(multiplier)


def attach_multiplier(algorithm: Algorithm) -> Algorithm:
    optimizers = [
        field.name
        for field in fields(algorithm)
        if isinstance(getattr(algorithm, field.name), optax.GradientTransformation)
    ]
    if not optimizers:
        raise ValueError(
            f"PBT scales the learning rate through a multiplier chained onto every "
            f"optax.GradientTransformation the algorithm holds, but "
            f"{type(algorithm).__name__} holds none."
        )
    return replace(
        algorithm,
        **{
            name: optax.chain(
                getattr(algorithm, name),
                optax.inject_hyperparams(scale_by_multiplier)(multiplier=1.0),
            )
            for name in optimizers
        },
    )


@struct.dataclass(frozen=True)
class PBTState(WrapperState):
    algorithm_state: PopulationState
    running: Array = struct.field(metadata={"axis": "data"})
    window: Array
    filled: Array
    replaced: Array
    updates: Array
    handled: Array
    seeded: Array
    key: Key
    step: Array


@dataclass
class PBT(Wrapper):
    members: int
    seeds: int
    window: int
    search: dict

    def __post_init__(self):
        self.search = {
            name: tuple(float(bound) for bound in bounds)
            for name, bounds in dict(self.search).items()
        }
        algorithm = self.algorithm
        if MULTIPLIER in self.search:
            algorithm = attach_multiplier(algorithm)
        injected = Injected(
            algorithm, names=tuple(name for name in self.search if name != MULTIPLIER)
        )
        self.algorithm = Population(injected, count=self.copies)

    @property
    def copies(self) -> int:
        return self.members * self.seeds

    def fitness(self, state: PBTState) -> tuple[Array, Array]:
        scores = state.window.mean(axis=1).reshape(self.members, self.seeds)
        ready = (state.filled >= self.window).reshape(self.members, self.seeds)
        return scores, ready.all(axis=1)

    def leader(self, state: PBTState) -> Array:
        scores, ready = self.fitness(state)
        mean = scores.mean(axis=1)
        return jnp.where(
            ready.any(),
            jnp.argmax(jnp.where(ready, mean, -jnp.inf)),
            jnp.argmax(mean),
        )

    def init(self, key: Key, timestep: Timestep) -> PBTState:
        population_key, search_key = jax.random.split(key)
        return PBTState(
            algorithm_state=self.algorithm.init(population_key, timestep),
            running=jnp.zeros(timestep.reward.shape, jnp.float32),
            window=jnp.zeros((self.copies, self.window), jnp.float32),
            filled=jnp.zeros(self.copies, jnp.int32),
            replaced=jnp.zeros(self.members, jnp.int32),
            updates=jnp.array(0, jnp.int32),
            handled=jnp.array(0, jnp.int32),
            seeded=jnp.array(False),
            key=search_key,
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
        )

    def synchronize(self, state: PBTState) -> PBTState:
        return state.replace(
            algorithm_state=self.algorithm.synchronize(
                state.algorithm_state.replace(step=state.step)
            )
        )

    def step(self, state: PBTState, key: Key, timestep: Timestep, temperature=1.0):
        state = self.synchronize(state)
        population_state, action, aux = self.algorithm.step(
            state.algorithm_state, key, timestep, temperature
        )
        return state.replace(algorithm_state=population_state), action, aux

    def update(self, state: PBTState, key: Key, transitions: Transition) -> PBTState:
        state = self.synchronize(state)
        return state.replace(
            algorithm_state=self.algorithm.update(
                state.algorithm_state, key, transitions
            ),
            updates=state.updates + 1,
        )
