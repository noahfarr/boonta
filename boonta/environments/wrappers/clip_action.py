import jax.numpy as jnp

from .wrapper import Wrapper


class ClipAction(Wrapper):

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        action_space = self._env.action_space()
        low = action_space.low if self.low is None else self.low
        high = action_space.high if self.high is None else self.high
        state, timestep = self._env.step(key, state, jnp.clip(action, low, high))
        return state, timestep.replace(action=action)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
