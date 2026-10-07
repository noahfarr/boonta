import math

import jax.numpy as jnp

from ..spaces import Space
from .transform_observation import TransformObservation


class FlattenObservation(TransformObservation):
    def __init__(self, env):
        super().__init__(env, jnp.ravel)

    def observation_space(self) -> Space:
        space = self._env.observation_space()
        return Space(
            shape=(math.prod(space.shape),),
            dtype=space.dtype,
            low=-jnp.inf,
            high=jnp.inf,
        )
