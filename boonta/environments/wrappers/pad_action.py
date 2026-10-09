from functools import partial

import jax.numpy as jnp

from ..spaces import Space
from .wrapper import Wrapper


class PadAction(Wrapper):
    def __init__(self, env, action_dim: int):
        super().__init__(env)
        self.action_dim = action_dim

    def pad(self, value, dim: int):
        value = jnp.asarray(value)
        if value.ndim == 0:
            return value
        return jnp.pad(value, (0, self.action_dim - dim))

    def action_space(self) -> Space:
        action_space = self._env.action_space()
        (dim,) = action_space.shape
        pad = partial(self.pad, dim=dim)
        return Space(
            shape=(self.action_dim,),
            dtype=action_space.dtype,
            low=pad(action_space.low),
            high=pad(action_space.high),
        )

    def step(self, key, state, action):
        (dim,) = self._env.action_space().shape
        state, timestep = self._env.step(key, state, action[..., :dim])
        return state, timestep.replace(action=action)

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
