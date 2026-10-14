from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, share, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key
from boonta.utils.typing import Environment


@dataclass
class ERD(Pit):
    environment: ArchiveAutoReset
    k: float = 4.0

    def select(self, archive_auto_reset_state: ArchiveAutoResetState, key: Key) -> Array:
        eligible_mask = self.environment.archive.eligible(
            archive_auto_reset_state.archive_state
        )
        num_placed = share(self.k, self.environment.num_envs)
        return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (num_placed,))

    def __call__(self, key: Key, state):
        select_key, assign_key = jax.random.split(key)

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                slots = self.select(node, select_key)
                return self.environment.update(node, assign_key, slots=slots)
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm, environment: Environment, k: float = 4.0, **kwargs
) -> tuple[Algorithm, Environment, ERD, Lap | None]:
    return algorithm, environment, ERD(unwrap(environment), k), None
