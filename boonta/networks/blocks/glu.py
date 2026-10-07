from collections.abc import Callable
from functools import partial

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from .block import Block


class GLU(Block):
    features: int
    expansion_factor: int = 4
    hidden_dim: int | None = None
    activation: Callable[[Array], Array] = nn.silu
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        hidden_dim = self.hidden_dim or self.features * self.expansion_factor
        gate = dense(hidden_dim, name="gate")(x)
        up = dense(hidden_dim, name="up")(x)
        x = dense(self.features, name="down")(self.activation(gate) * up)
        return None, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return None
