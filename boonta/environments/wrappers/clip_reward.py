from functools import partial

import jax.numpy as jnp

from .transform_reward import TransformReward


class ClipReward(TransformReward):

    def __init__(self, env, low=None, high=None):
        super().__init__(env, fn=partial(jnp.clip, min=low, max=high))
