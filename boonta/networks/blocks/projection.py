from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from .block import Block


class Projection(Block):

    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    activation: Callable[[Array], Array] | None = None

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        x = nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.activation is not None:
            x = self.activation(x)
        return None, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return None
