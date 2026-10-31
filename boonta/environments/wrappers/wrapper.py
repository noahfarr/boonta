from typing import Any

from flax import struct

from ..environment import Environment


@struct.dataclass
class WrapperState:
    env_state: Any

    def __getattr__(self, name):
        # Only called for missing attributes. While copy or pickle builds a new object, `env_state` itself is missing,
        # and forwarding it (or a dunder lookup such as __setstate__) would recurse without end.
        if name == "env_state" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class Wrapper(Environment):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        # See WrapperState.__getattr__: `_env` is missing on an object that copy or pickle is still building.
        if name == "_env" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
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

    def update(self, state, key, **kwargs):
        return state.replace(env_state=self._env.update(state.env_state, key, **kwargs))

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()

    def action_mask(self, state):
        return self._env.action_mask(state.env_state)

    def observe(self, state):
        return self._env.observe(state.env_state)

    def time_limit(self):
        return self._env.time_limit()

    def render(self, state, **kwargs):
        return self._env.render(state, **kwargs)

    def close(self, state):
        self._env.close(getattr(state, "unwrapped", state))

    def wraps(self, cls: type) -> bool:
        env = self._env
        while isinstance(env, Wrapper):
            if isinstance(env, cls):
                return True
            env = env._env
        return isinstance(env, cls)
