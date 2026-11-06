from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key
from boonta.utils.typing import Environment


@dataclass
class ERD(Pit):
    environment: ArchiveAutoReset
    k: float = 4.0
    restart_at: str = "rollout"

    def __post_init__(self):
        assert isinstance(
            self.environment, ArchiveAutoReset
        ), "ERD restarts from an ArchiveAutoReset, and the environment wraps none"
        assert self.k >= 1.0, f"k must be at least 1, got {self.k}"
        assert self.restart_at in ("rollout", "episode"), self.restart_at

    @property
    def num_placed(self) -> int:
        return int(round((1.0 - 1.0 / self.k) * self.environment.num_envs))

    def select(self, archive_auto_reset_state: ArchiveAutoResetState, key: Key) -> Array:
        eligible_mask = self.environment.archive.eligible(
            archive_auto_reset_state.archive_state
        )
        return pick(key, jnp.where(eligible_mask, 0.0, -jnp.inf), (self.num_placed,))

    def __call__(self, key: Key, state):
        select_key, assign_key = jax.random.split(key)

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                slots = self.select(node, select_key)
                return self.environment.update(
                    node, assign_key, slots=slots, preempt=self.restart_at == "rollout"
                )
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm,
    environment: Environment,
    k: float = 4.0,
    restart_at: str = "rollout",
    **kwargs,
) -> tuple[Algorithm, Environment, ERD, Lap | None]:
    return algorithm, environment, ERD(unwrap(environment), k, restart_at), None
