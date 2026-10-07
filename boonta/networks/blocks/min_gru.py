from functools import partial

import flax.linen as nn
import jax.numpy as jnp
from boonta.utils.typing import Array, Dtype, Key

from .ssm import SSMCellBase


class MinGRUCell(SSMCellBase):
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        dense = partial(
            nn.Dense,
            self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        z = nn.sigmoid(dense(name="update_gate")(x))

        x = dense(name="candidate")(x)
        return 1 - z, z * jnp.where(x >= 0, x + 0.5, nn.sigmoid(x))

    def output(self, carries: Array, x: Array) -> Array:
        return jnp.asarray(carries, self.dtype)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Array:
        *batch_size, _ = input_shape
        return jnp.zeros((*batch_size, self.features))
