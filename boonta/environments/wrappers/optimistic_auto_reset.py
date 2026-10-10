import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, PyTree, Timestep, broadcast

from .auto_reset import AutoReset


class OptimisticAutoReset(AutoReset):

    def __init__(self, env, num_envs: int, ratio: int):
        super().__init__(env, num_envs=num_envs)
        assert num_envs % ratio == 0
        self.ratio = ratio

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Timestep]:
        step_key, reset_key, choice_key = jax.random.split(key, 3)
        state, timestep = super().step(step_key, state, action)

        num_resets = self.num_envs // self.ratio
        fresh, initial = jax.vmap(self._env.init)(
            jax.random.split(reset_key, num_resets)
        )

        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        chosen = jax.random.choice(
            choice_key,
            self.num_envs,
            shape=(num_resets,),
            replace=False,
            p=jnp.where(
                done.any(),
                done / jnp.maximum(done.sum(), 1),
                1.0 / self.num_envs,
            ),
        )
        slots = jnp.arange(self.num_envs) // self.ratio
        slots = slots.at[chosen].set(jnp.arange(num_resets))

        def select(fresh_leaf, leaf):
            return jnp.where(
                broadcast(done, leaf), jnp.take(fresh_leaf, slots, axis=0), leaf
            )

        state = jax.tree.map(select, fresh, state)
        obs = jax.tree.map(select, initial.obs, timestep.obs)
        return state, timestep.replace(obs=obs)
