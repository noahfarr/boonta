from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers import archive_auto_reset
from boonta.environments.wrappers.archive_auto_reset import (locate, pick, plant,
                                                             share, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


@dataclass
class Selector(archive_auto_reset.Selector):
    k: float = 4.0

    def select(self, state, wrapper, archive_auto_reset_state, key):
        mask = wrapper.archive.eligible(archive_auto_reset_state.archive_state)
        placed = share(self.k, wrapper.num_envs)
        return state, pick(key, jnp.where(mask, 0.0, -jnp.inf), (placed,))


def pit(
    algorithm: Algorithm, environment: Environment, seed: int = 0, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    wrapper = unwrap(environment)

    def place(state, transitions):
        key = jax.random.fold_in(
            jax.random.key(seed), state.algorithm_state.step.astype(jnp.uint32)
        )
        placed = wrapper.place(locate(state.environment_state), key, transitions)
        return state.replace(environment_state=plant(state.environment_state, placed))

    return algorithm, environment, place, lambda state: state
