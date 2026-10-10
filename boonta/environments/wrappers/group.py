from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class GroupState(WrapperState):
    step: Array


class Group(Wrapper):
    def __init__(self, env, num_steps: int, group_size: int = 1):
        super().__init__(env)
        self.num_steps = num_steps
        self.group_size = group_size

    def share(self, tree: PyTree) -> PyTree:
        def leader(leaf):
            groups = jnp.reshape(leaf, (-1, self.group_size, *jnp.shape(leaf)[1:]))
            return jnp.reshape(
                jnp.broadcast_to(groups[:, :1], jnp.shape(groups)), jnp.shape(leaf)
            )

        return jax.tree.map(leader, tree)

    def start(self, key: Key) -> tuple[PyTree, Timestep]:
        env_state, timestep = self._env.init(key)
        return self.share(env_state), timestep.replace(obs=self.share(timestep.obs))

    def init(self, key: Key) -> tuple[GroupState, Timestep]:
        env_state, timestep = self.start(key)
        return GroupState(env_state, jnp.zeros(self.num_envs, jnp.int32)), timestep

    def step(self, key: Key, state: GroupState, action: Array) -> tuple[GroupState, Timestep]:
        step_key, reset_key = jax.random.split(key)
        env_state, timestep = self._env.step(step_key, state.env_state, action)
        step = state.step + 1

        def restart(_):
            fresh, initial = self.start(reset_key)
            return fresh, initial.obs

        boundary = (step % self.num_steps == 0).all()
        env_state, obs = jax.lax.cond(
            boundary, restart, lambda _: (env_state, timestep.obs), None
        )
        truncated = timestep.truncated | (boundary & ~timestep.terminated)
        return GroupState(env_state, step), timestep.replace(obs=obs, truncated=truncated)
