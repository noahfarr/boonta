from collections.abc import Callable

from .wrapper import Wrapper


class TransformReward(Wrapper):
    def __init__(self, env, fn: Callable):
        super().__init__(env)
        self.fn = fn

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        return state, timestep.replace(reward=self.fn(timestep.reward))

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
