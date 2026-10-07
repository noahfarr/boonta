from collections.abc import Sequence
from functools import reduce

import flax.linen as nn

from boonta.utils.typing import Array, Carry, Key

from .block import Block, broadcast_carry


class Stack(Block):
    blocks: Sequence[Block]

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        def step(state: tuple[Carry, Array], inputs) -> tuple[Carry, Array]:
            carry, x = state
            block_carry, block = inputs
            block_carry, x = block(block_carry, x, done)
            return (*carry, block_carry), x

        carries = broadcast_carry(carry, len(self.blocks))
        return reduce(step, zip(carries, self.blocks, strict=True), ((), x))

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return tuple(block.initialize_carry(key, input_shape) for block in self.blocks)
