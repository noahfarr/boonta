import flax.linen as nn

from boonta.utils.typing import Array, Carry, Key

from .block import Block


class Residual(Block):
    blocks: Block

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        skip = x
        carry, x = self.blocks(carry, x, done)
        return carry, skip + x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.blocks.initialize_carry(key, input_shape)
