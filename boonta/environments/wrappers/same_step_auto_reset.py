from typing import Any

import jax

from boonta.utils import Array, Key, Timestep

from .auto_reset import AutoReset


class SameStepAutoReset(AutoReset):

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        def restart(key, state, action):
            step_key, reset_key = jax.random.split(key)
            env_state, timestep = self._env.step(step_key, state, action)
            initial_env_state, initial_timestep = self._env.init(reset_key)
            done = timestep.done.all()

            def select(initial_leaf, leaf):
                return jax.lax.select(done, initial_leaf, leaf)

            env_state = jax.tree.map(select, initial_env_state, env_state)
            obs = jax.tree.map(select, initial_timestep.obs, timestep.obs)
            return env_state, timestep.replace(obs=obs)

        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(restart)(keys, state, action)
