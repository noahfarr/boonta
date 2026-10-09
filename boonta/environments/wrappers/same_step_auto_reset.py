from typing import Any

import jax

from boonta.utils import Array, Key, PyTree, Timestep

from .wrapper import Wrapper


class SameStepAutoReset(Wrapper):

    def step(
        self, key: Key, state: Any, action: Array
    ) -> tuple[Any, Timestep]:
        step_key, reset_key = jax.random.split(key)

        env_state, timestep = self._env.step(step_key, state, action)
        initial_env_state, initial_timestep = self._env.init(reset_key)

        done = timestep.done.all()

        def select(initial_leaf, leaf):
            return jax.lax.select(done, initial_leaf, leaf)

        env_state = jax.tree.map(select, initial_env_state, env_state)
        obs = jax.tree.map(select, initial_timestep.obs, timestep.obs)
        return env_state, timestep.replace(obs=obs)

    def update(self, state: PyTree, key: Key, **kwargs) -> PyTree:
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state: PyTree) -> Array | None:
        return self._env.action_mask(state)

    def observe(self, state: PyTree) -> PyTree:
        return self._env.observe(state)
