import flax.linen as nn

from boonta.utils.typing import Array, Carry, Key

from .block import Block


class Identity(Block):
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        return None, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return None
