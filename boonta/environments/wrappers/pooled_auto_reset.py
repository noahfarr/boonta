import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, PyTree, Timestep, broadcast

from .auto_reset import AutoReset


class PooledAutoReset(AutoReset):

    def __init__(self, env, num_envs: int, size: int, seed: int = 0):
        super().__init__(env, num_envs=num_envs)
        assert size > 0
        self.size = size
        self.pool, self.initial = jax.jit(jax.vmap(env.init))(
            jax.random.split(jax.random.key(seed), size)
        )

    def draw(self, key: Key) -> tuple[PyTree, Timestep]:
        drawn = jax.random.randint(key, (self.num_envs,), 0, self.size)

        def take(leaf):
            return jnp.take(leaf, drawn, axis=0)

        return jax.tree.map(take, self.pool), jax.tree.map(take, self.initial)

    def init(self, key: Key) -> tuple[PyTree, Timestep]:
        return self.draw(key)

    def step(self, key: Key, state: PyTree, action: Array) -> tuple[PyTree, Timestep]:
        step_key, draw_key = jax.random.split(key)
        state, timestep = super().step(step_key, state, action)
        fresh, initial = self.draw(draw_key)
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)

        def select(fresh_leaf, leaf):
            return jnp.where(broadcast(done, leaf), fresh_leaf, leaf)

        state = jax.tree.map(select, fresh, state)
        obs = jax.tree.map(select, initial.obs, timestep.obs)
        return state, timestep.replace(obs=obs)
