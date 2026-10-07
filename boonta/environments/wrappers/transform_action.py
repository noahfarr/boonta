from collections.abc import Callable

from .wrapper import Wrapper


class TransformAction(Wrapper):
    def __init__(self, env, fn: Callable):
        super().__init__(env)
        self.fn = fn

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, self.fn(action))
        return state, timestep.replace(action=action)

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)
