from typing import Any

from flax import struct

from ..environment import Environment


@struct.dataclass
class WrapperState:
    env_state: Any

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class Wrapper(Environment):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self._env)

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def init(self, key):
        return self._env.init(key)

    def step(self, key, state, action):
        return self._env.step(key, state, action)

    def update(self, state, **kwargs):
        return state.replace(env_state=self._env.update(state.env_state, **kwargs))

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()

    def action_mask(self, state):
        return self._env.action_mask(state.env_state)

    def time_limit(self):
        return self._env.time_limit()

    def render(self, state):
        return self._env.render(state)

    def close(self, state):
        self._env.close(getattr(state, "unwrapped", state))

    def wraps(self, cls: type) -> bool:
        env = self._env
        while isinstance(env, Wrapper):
            if isinstance(env, cls):
                return True
            env = env._env
        return isinstance(env, cls)
