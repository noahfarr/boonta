from collections.abc import Callable

from .wrapper import Wrapper


class TransformObservation(Wrapper):
    def __init__(self, env, fn: Callable):
        super().__init__(env)
        self.fn = fn

    def init(self, key):
        state, timestep = self._env.init(key)
        return state, timestep.replace(obs=self.fn(timestep.obs))

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        return state, timestep.replace(obs=self.fn(timestep.obs))

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)
