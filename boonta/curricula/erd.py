import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (locate, pick, plant,
                                                             share, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


def select(wrapper, archive_auto_reset_state, key, k: float = 4.0):
    eligible_mask = wrapper.archive.eligible(archive_auto_reset_state.archive_state)
    num_placed = share(k, wrapper.num_envs)
    return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (num_placed,))


def pit(
    algorithm: Algorithm, environment: Environment, k: float = 4.0, **kwargs
) -> tuple[Algorithm, Environment, Pit, Lap]:
    wrapper = unwrap(environment)

    def assign(key, state, transitions):
        select_key, assign_key = jax.random.split(key)
        archived = locate(state.environment_state)
        slots = select(wrapper, archived, select_key, k)
        archived = wrapper.update(archived, assign_key, slots=slots)
        return state.replace(environment_state=plant(state.environment_state, archived))

    return algorithm, environment, assign, lambda key, state: state
