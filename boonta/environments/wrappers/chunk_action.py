import math

import jax
import jax.numpy as jnp

from ..spaces import Space
from .wrapper import Wrapper


class ChunkAction(Wrapper):
    def __init__(self, env, size: int):
        super().__init__(env)
        self.size = size

    def init(self, key):
        state, timestep = self._env.init(key)
        action = jnp.broadcast_to(timestep.action, (self.size, *jnp.shape(timestep.action)))
        return state, timestep.replace(action=action)

    def step(self, key, state, action):
        keys = jax.random.split(key, self.size)
        state, timestep = self._env.step(keys[0], state, action[0])

        def act(carry, inputs):
            state, timestep, reward = carry
            key, action = inputs
            done = timestep.terminated | timestep.truncated
            next_state, next_timestep = self._env.step(key, state, action)
            state, timestep = jax.tree.map(
                lambda current, following: jnp.where(done, current, following),
                (state, timestep),
                (next_state, next_timestep),
            )
            reward = reward + jnp.where(done, 0.0, next_timestep.reward)
            return (state, timestep, reward), None

        (state, timestep, reward), _ = jax.lax.scan(
            act, (state, timestep, timestep.reward), (keys[1:], action[1:])
        )
        return state, timestep.replace(action=action, reward=reward)

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_space(self):
        space = self._env.action_space()
        return Space((self.size, *space.shape), space.dtype, space.low, space.high)

    def action_mask(self, state):
        mask = self._env.action_mask(state)
        return jnp.broadcast_to(mask, (self.size, *mask.shape))

    def observe(self, state):
        return self._env.observe(state)

    def time_limit(self):
        return math.ceil(self._env.time_limit() / self.size)
