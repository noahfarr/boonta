import jax.numpy as jnp
import numpy as np

from ..spaces import Space
from .wrapper import Wrapper


class Prompt(Wrapper):
    def __init__(self, env, prompt, pad: int):
        super().__init__(env)
        self._prompt = jnp.asarray(np.asarray(prompt, np.int32))
        self._pad = pad

    def init(self, key):
        state, timestep = self._env.init(key)
        return state, timestep.replace(obs=jnp.concatenate([self._prompt, timestep.obs]))

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        pads = jnp.full_like(self._prompt, self._pad)
        return state, timestep.replace(obs=jnp.concatenate([pads, timestep.obs]))

    def observation_space(self) -> Space:
        space = self._env.observation_space()
        (length,) = space.shape
        return Space(
            shape=(len(self._prompt) + length,),
            dtype=space.dtype,
            low=space.low,
            high=space.high,
        )

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        raise NotImplementedError
