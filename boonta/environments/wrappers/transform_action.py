from collections.abc import Callable

from .wrapper import Wrapper


class TransformAction(Wrapper):
    def __init__(self, env, fn: Callable):
        super().__init__(env)
        self.fn = fn

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, self.fn(action))
        return state, timestep.replace(action=action)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
