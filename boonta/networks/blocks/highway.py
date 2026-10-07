import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from .block import Block


class Highway(Block):
    blocks: Block
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        *_, features = x.shape
        gate = nn.sigmoid(
            nn.Dense(
                features,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            )(x)
        )
        skip = x
        carry, x = self.blocks(carry, x, done)
        return carry, gate * x + (1 - gate) * skip

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.blocks.initialize_carry(key, input_shape)
