import flax.linen as nn

from boonta.utils.typing import Array, Carry, Key


class Block(nn.Module):
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        raise NotImplementedError


def broadcast_carry(carry: Carry, length: int) -> Carry:
    if carry is None:
        carry = [None] * length
    return carry
