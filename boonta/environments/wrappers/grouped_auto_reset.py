from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class GroupedAutoResetState(WrapperState):
    step: Array


class GroupedAutoReset(Wrapper):
    def __init__(self, env, num_steps: int, group_size: int = 1):
        super().__init__(env)
        self.num_steps = num_steps
        self.group_size = group_size

    def init(self, key: Key, group_size: int | None = None) -> tuple[Any, Timestep]:
        env_state, timestep = self._env.init(
            key, group_size=self.group_size if group_size is None else group_size
        )
        state = GroupedAutoResetState(env_state, jnp.zeros(self.num_envs, jnp.int32))
        return state, timestep

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        step_key, reset_key = jax.random.split(key)
        env_state, timestep = self._env.step(step_key, state.env_state, action)
        step = state.step + 1

        def restart(_):
            fresh, initial = self._env.init(reset_key, group_size=self.group_size)
            return fresh, initial.obs

        boundary = (step % self.num_steps == 0).all()
        env_state, obs = jax.lax.cond(
            boundary, restart, lambda _: (env_state, timestep.obs), None
        )
        truncated = timestep.truncated | (boundary & ~timestep.terminated)
        return (
            GroupedAutoResetState(env_state, step),
            timestep.replace(obs=obs, truncated=truncated),
        )
