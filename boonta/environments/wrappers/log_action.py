from collections.abc import Iterable
from typing import Any

import lox

from boonta.utils import Array, Key

from .wrapper import Wrapper


class LogAction(Wrapper):
    def __init__(self, env, tags: Iterable[str] = ("evaluation",)):
        super().__init__(env)
        self.tags = tags

    def step(
        self, key: Key, state: Any, action: Array
    ) -> tuple[Any, Any]:
        lox.log({"action": action}, tags=self.tags)
        return self._env.step(key, state, action)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
