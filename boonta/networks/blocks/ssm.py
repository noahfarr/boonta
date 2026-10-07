import jax
import jax.numpy as jnp
from flax import linen as nn
from boonta.utils import broadcast
from boonta.utils.typing import Array, Carry, Key

from .block import Block
from ..kernels import get_scan_implementation


def binary_operator(
    lhs: tuple[Array, Array], rhs: tuple[Array, Array]
) -> tuple[Array, Array]:
    a_i, b_i = lhs
    a_j, b_j = rhs
    return a_j * a_i, a_j * b_i + b_j


class SSMCellBase(nn.Module):
    def __call__(self, x: Array) -> tuple[Array, Array]:
        raise NotImplementedError

    def output(self, carries: Array, x: Array) -> Array:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Array:
        raise NotImplementedError


class SSM(Block):
    cell: SSMCellBase

    @nn.compact
    def __call__(self, carry: Carry, inputs: Array, done: Array) -> tuple[Carry, Array]:
        batch_size, _, *features = inputs.shape
        initial_carry = self.cell.initialize_carry(
            jax.random.key(0), (batch_size, *features)
        )
        if carry is None:
            carry = initial_carry

        a, b = nn.remat(lambda cell, x: cell(x))(self.cell, inputs)
        scan_dtype = jnp.result_type(a, b)
        a, b = a.astype(scan_dtype), b.astype(scan_dtype)
        mask = broadcast(done, b)
        b = b + mask * a * jnp.expand_dims(initial_carry, 1).astype(scan_dtype)
        a = (1 - mask) * a

        scan = get_scan_implementation()
        carries = scan(a, b, carry.astype(scan_dtype))
        return (
            jnp.take(carries, -1, axis=1).astype(carry.dtype),
            self.cell.output(carries, inputs),
        )

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
