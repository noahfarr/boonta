from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (locate, pick, plant,
                                                             share, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


@dataclass
class Selector:
    k: float = 4.0

    def select(self, wrapper, archive_auto_reset_state, key):
        eligible_mask = wrapper.archive.eligible(archive_auto_reset_state.archive_state)
        num_placed = share(self.k, wrapper.num_envs)
        return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (num_placed,))


def pit(
    algorithm: Algorithm, environment: Environment, selector: Selector, seed: int = 0, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    wrapper = unwrap(environment)

    def assign(state, transitions):
        key = jax.random.fold_in(
            jax.random.key(seed), state.algorithm_state.step.astype(jnp.uint32)
        )
        select_key, assign_key = jax.random.split(key)
        archived = locate(state.environment_state)
        slots = selector.select(wrapper, archived, select_key)
        archived = wrapper.update(archived, assign_key, slots=slots)
        return state.replace(environment_state=plant(state.environment_state, archived))

    return algorithm, environment, assign, lambda state: state
