import jax
import jax.numpy as jnp
import lox

from boonta.utils import PyTree

from .wrapper import Wrapper


class MaskObservation(Wrapper):
    def __init__(self, env, mask: PyTree):
        super().__init__(env)
        self.mask = mask
        leaves = jax.tree.leaves(mask)
        total = sum(leaf.size for leaf in leaves)
        masked = sum((leaf == 0).sum() for leaf in leaves)
        self.mask_rate = jnp.asarray(masked / total, dtype=jnp.float32)

    def init(self, key):
        state, timestep = self._env.init(key)
        obs = jax.tree.map(lambda o, m: o * m, timestep.obs, self.mask)
        return state, timestep.replace(obs=obs)

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        lox.log({"mask_observation/mask_rate": self.mask_rate})
        obs = jax.tree.map(lambda o, m: o * m, timestep.obs, self.mask)
        return state, timestep.replace(obs=obs)

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)
