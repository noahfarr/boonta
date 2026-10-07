import jax.numpy as jnp

from ..spaces import Space
from .wrapper import Wrapper


class PadObservation(Wrapper):
    def __init__(self, env, key: str, obs_dim: int):
        super().__init__(env)
        self.key = key
        self.obs_dim = obs_dim

    def pad(self, value, native_dim: int):
        value = jnp.asarray(value)
        if value.ndim == 0:
            return value
        return jnp.pad(value, (0, self.obs_dim - native_dim))

    def observation_space(self) -> dict:
        observation_space = self._env.observation_space()
        space = observation_space[self.key]
        (native_dim,) = space.shape
        return observation_space | {
            self.key: Space(
                shape=(self.obs_dim,),
                dtype=space.dtype,
                low=self.pad(space.low, native_dim),
                high=self.pad(space.high, native_dim),
            )
        }

    def transform(self, obs):
        native_dim = obs[self.key].shape[-1]
        return obs | {self.key: self.pad(obs[self.key], native_dim)}

    def init(self, key):
        state, timestep = self._env.init(key)
        return state, timestep.replace(obs=self.transform(timestep.obs))

    def step(self, key, state, action):
        state, timestep = self._env.step(key, state, action)
        return state, timestep.replace(obs=self.transform(timestep.obs))

    def update(self, state, **kwargs):
        return self._env.update(state, **kwargs)
